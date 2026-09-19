"""Loss estimation, persistence baseline, and pretrain/finetune training loops."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .dataset import FinetuneBatcher, PretrainBatcher
from .model import DecoderTransformer


@torch.no_grad()
def estimate_loss(model: DecoderTransformer, batcher: PretrainBatcher, batch_size: int,
                   split: str, iters: int = 100) -> float:
    model.eval()
    eval_loss = torch.zeros(iters)
    for step in range(iters):
        tx, ty, tm = batcher.get_batch(batch_size, split)
        _, loss = model.pretraining_head(tx, ty, tm)
        eval_loss[step] = loss
    return eval_loss.mean().item()


@torch.no_grad()
def estimate_loss_bolus_head(model: DecoderTransformer, batcher: FinetuneBatcher, batch_size: int,
                              split: str, pos_weight: torch.Tensor, eval_iters: int = 100):
    model.eval()
    eval_loss = torch.zeros(eval_iters)
    all_probs, all_labels = [], []
    offset = model.config.eval_offset
    for step in range(eval_iters):
        tx, ty = batcher.get_batch(batch_size, split)
        logits, loss = model.finetuning_head(tx, ty, pos_weight=pos_weight)
        eval_loss[step] = loss
        all_probs.append(torch.sigmoid(logits).flatten())
        all_labels.append(ty[:, offset:].flatten())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    auroc = roc_auc_score(labels, probs) if labels.sum() > 0 else float("nan")
    ap = average_precision_score(labels, probs) if labels.sum() > 0 else float("nan")
    return eval_loss.mean().item(), auroc, ap


def weighted_persistence_mse_masked(y: np.ndarray, mask: np.ndarray, weights: torch.Tensor, horizon: int) -> float:
    """Baseline: predict each signal `horizon` steps ahead as unchanged from now."""
    persistence_pred = y[:-horizon]
    target = y[horizon:]
    m = mask[horizon:]
    w = weights.numpy()
    sq_err = (persistence_pred - target) ** 2 * m
    return (sq_err * w).sum() / (m * w).sum()


def pretrain(model, optimizer, batcher: PretrainBatcher, batch_size: int, epochs: int,
             log_every: int = 100, show_every: int = 200, verbose: bool = True):
    train_loss_arr = np.array([])
    test_loss_arr = np.array([])
    for step in range(1, epochs + 1):
        bx, by, bm = batcher.get_batch(batch_size, "train")
        logits, loss = model.pretraining_head(bx, by, bm)
        train_loss_arr = np.append(train_loss_arr, loss.item())
        if step % log_every == 0:
            test_loss = estimate_loss(model, batcher, batch_size, "test")
            test_loss_arr = np.append(test_loss_arr, test_loss)
            if verbose and step % show_every == 0:
                print(f"  [pretrain] step {step}: train loss {loss.item():.4f}, test loss {test_loss:.4f}")
            model.train()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model, train_loss_arr, test_loss_arr


def finetune(model, optimizer, batcher: FinetuneBatcher, batch_size: int, pos_weight: torch.Tensor,
             epochs: int, log_every: int = 100, show_every: int = 200, verbose: bool = True):
    train_loss_arr = np.array([])
    test_loss_arr = np.array([])
    for step in range(1, epochs + 1):
        bx, by = batcher.get_batch(batch_size, "train")
        logits, loss = model.finetuning_head(bx, by, pos_weight=pos_weight)
        train_loss_arr = np.append(train_loss_arr, loss.item())
        if step % log_every == 0:
            test_loss, auroc, ap = estimate_loss_bolus_head(model, batcher, batch_size, "test", pos_weight)
            test_loss_arr = np.append(test_loss_arr, test_loss)
            if verbose and step % show_every == 0:
                print(f"  [finetune] step {step}: train loss {loss.item():.4f}, test loss {test_loss:.4f}, "
                      f"auroc {auroc:.4f}, ap {ap:.4f}")
            model.train()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model, train_loss_arr, test_loss_arr
