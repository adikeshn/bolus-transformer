import copy
import random
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .dataset import WindowSampler, sweep_windows
from .metrics import ranking_metrics

Arrays = Dict[str, np.ndarray]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def _sweep(model, x: np.ndarray, config: Config, fn: Callable) -> np.ndarray:
    windows = sweep_windows(len(x), config.context_length, config.eval_offset)
    if not windows:
        return np.zeros((0,))
    model.eval()
    c = config.context_length
    batch = torch.from_numpy(np.stack([x[s:s + c] for s, _ in windows])).to(DEVICE)
    out = fn(batch).cpu().numpy()
    return np.concatenate([out[i, keep:] for i, (_, keep) in enumerate(windows)])


def forecast_rows(model, a: Arrays, config: Config):
    n, h, off = len(a["x"]), config.horizon, config.eval_offset
    pred = _sweep(model, a["x"][:n - h], config, model.forecast)
    return pred, a["y"][off + h:n], a["mask"][off + h:n]


def bolus_probs(model, a: Arrays, config: Config) -> np.ndarray:
    return 1 / (1 + np.exp(-_sweep(model, a["x"], config, model.bolus_logits)))


def weighted_mse(pred, target, mask, weights: np.ndarray):
    if len(pred) == 0:
        return 0.0, 0.0
    w = mask * weights
    return float(((pred - target) ** 2 * w).sum()), float(w.sum())


def pooled_forecast_mse(model, arrays: List[Arrays], config: Config) -> float:
    w = config.weights_tensor().numpy()
    num = den = 0.0
    for a in arrays:
        n_, d_ = weighted_mse(*forecast_rows(model, a, config), w)
        num, den = num + n_, den + d_
    return num / max(den, 1e-8)


def pooled_persistence_mse(arrays: List[Arrays], config: Config) -> float:
    w = config.weights_tensor().numpy()
    h, off = config.horizon, config.eval_offset
    num = den = 0.0
    for a in arrays:
        n = len(a["x"])
        if n - h < config.context_length:
            continue
        n_, d_ = weighted_mse(a["y"][off:n - h], a["y"][off + h:n], a["mask"][off + h:n], w)
        num, den = num + n_, den + d_
    return num / max(den, 1e-8)


def pooled_ap(model, arrays: List[Arrays], config: Config) -> float:
    probs = np.concatenate([bolus_probs(model, a, config) for a in arrays])
    labels = np.concatenate([a["labels"][config.eval_offset:] for a in arrays])
    return ranking_metrics(probs, labels)[1]


def _train(model, config: Config, lr: float, steps: int, step_loss: Callable, validate: Callable,
           higher_is_better: bool, name: str, verbose: bool):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sign = 1 if higher_is_better else -1
    best_score, best_state, bad, history = -np.inf, copy.deepcopy(model.state_dict()), 0, []
    for step in range(1, steps + 1):
        model.train()
        loss = step_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config.eval_every == 0:
            val = validate()
            history.append({"step": step, "train_loss": loss.item(), "val": val})
            if verbose:
                print(f"  [{name}] step {step}: train loss {loss.item():.4f}, val {val:.4f}")
            if sign * val > best_score:
                best_score, best_state, bad = sign * val, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= config.patience:
                    break
    model.load_state_dict(best_state)
    return model, history


def pretrain(model, train: Dict[str, Arrays], val: List[Arrays], config: Config, verbose=True):
    sampler = WindowSampler(train, config.context_length, config.horizon)
    weights, off = config.weights_tensor().to(DEVICE), config.eval_offset

    def step_loss():
        x, y, m = (t.to(DEVICE) for t in sampler.pretrain_batch(config.batch_size))
        pred = model.forecast(x)[:, off:]
        w = m[:, off:] * weights
        return ((pred - y[:, off:]) ** 2 * w).sum() / w.sum().clamp(min=1e-8)

    return _train(model, config, config.pretrain_lr, config.pretrain_steps, step_loss,
                  lambda: pooled_forecast_mse(model, val, config), False, "pretrain", verbose)


def finetune(model, train: Dict[str, Arrays], val: List[Arrays], config: Config, verbose=True):
    sampler = WindowSampler(train, config.context_length)
    labels = np.concatenate([a["labels"] for a in train.values()])
    pos_weight = torch.tensor((len(labels) - labels.sum()) / max(labels.sum(), 1.0),
                              dtype=torch.float32, device=DEVICE)
    off = config.eval_offset

    def step_loss():
        x, y = (t.to(DEVICE) for t in sampler.finetune_batch(config.batch_size))
        logits = model.bolus_logits(x)[:, off:]
        return F.binary_cross_entropy_with_logits(logits, y[:, off:], pos_weight=pos_weight)

    return _train(model, config, config.finetune_lr, config.finetune_steps, step_loss,
                  lambda: pooled_ap(model, val, config), True, "finetune", verbose)
