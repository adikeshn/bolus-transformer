"""Leave-one-subject-out (LOSO) cross-validation over the OhioT1DM cohort."""
from typing import List

import numpy as np
import pandas as pd
import torch

from .config import SIGNALS, SIGNALS_FOR_LOSS, Config
from .dataset import FinetuneBatcher, PretrainBatcher
from .model import DecoderTransformer
from .preprocessing import gen_finetuning_data, get_data, preprocess_data
from .training import estimate_loss, estimate_loss_bolus_head, finetune, pretrain, weighted_persistence_mse_masked


def run_fold(held_out_subject: int, all_subjects: List[int], config: Config = Config(),
             data_dir: str = "bolus_data", verbose: bool = True):
    """Trains and evaluates one LOSO fold: pretrains on every subject but
    `held_out_subject`, then fine-tunes and evaluates bolus-timing classification
    on the held-out subject. Returns (result_dict, loss_histories)."""
    subject_ids = [s for s in all_subjects if s != held_out_subject]
    subject_ids_test = [held_out_subject]

    if verbose:
        print(f"\n=== Fold: holding out subject {held_out_subject} ===")
        print(f"training on {subject_ids}")

    # --- load + align pretraining data, then fit normalization on the training subjects only ---
    training_concat = np.array([])
    subject_dfs = {}

    for subject_num in subject_ids + subject_ids_test:
        x, _, mask = get_data(subject_num, "training", config.iob_peak, config.iob_dia, data_dir)
        subject_dfs[subject_num] = (x, mask)
        if subject_num in subject_ids:
            chunk = subject_dfs[subject_num][0][SIGNALS].to_numpy()
            training_concat = chunk if training_concat.shape[0] == 0 else np.concatenate((training_concat, chunk))

    means, stds = {}, {}
    for index, signal in enumerate(SIGNALS[:-2]):
        means[signal] = np.nanmean(training_concat[:, index])
        stds[signal] = np.nanstd(training_concat[:, index])

    training_d, testing_d = {}, {}
    for subject_num in subject_ids + subject_ids_test:
        x_df, mask = subject_dfs[subject_num]
        x_p, y_p = preprocess_data(x_df, SIGNALS, means, stds)
        data = (
            x_p[SIGNALS].to_numpy().astype(np.float32),
            y_p.to_numpy().astype(np.float32),
            mask,
        )
        if subject_num in subject_ids:
            training_d[subject_num] = data
        else:
            testing_d[subject_num] = data

    weights = config.weights_tensor()
    persistence_baseline = weighted_persistence_mse_masked(
        testing_d[held_out_subject][1], testing_d[held_out_subject][2], weights, horizon=config.horizon
    )
    if verbose:
        print(f"30-min weighted persistence baseline ({held_out_subject}): {persistence_baseline:.4f}")

    # --- pretraining ---
    pretrain_batcher = PretrainBatcher(training_d, testing_d, config.context_length, config.horizon)
    model = DecoderTransformer(config, n_signals_for_loss=len(SIGNALS_FOR_LOSS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_lr)
    model, pt_train_arr, pt_test_arr = pretrain(
        model, optimizer, pretrain_batcher, config.batch_size, epochs=config.pretrain_epochs,
        show_every=500 if verbose else config.pretrain_epochs + 1, verbose=verbose,
    )
    pretrain_test_loss = estimate_loss(model, pretrain_batcher, config.batch_size, "test")
    pretrain_train_loss = estimate_loss(model, pretrain_batcher, config.batch_size, "train")

    # --- fine-tuning data ---
    bolus_train, bolus_test = gen_finetuning_data(
        subject_ids, subject_ids_test, means, stds, config.iob_peak, config.iob_dia,
        config.finetune_label_window, data_dir,
    )
    finetune_batcher = FinetuneBatcher(bolus_train, bolus_test, config.context_length)

    all_train_labels = np.concatenate([bolus_train[s][1] for s in subject_ids])
    num_pos = all_train_labels.sum()
    num_neg = len(all_train_labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos])
    if verbose:
        print(f"positive rate: {num_pos / len(all_train_labels):.4f}, pos_weight: {pos_weight.item():.2f}")

    # --- fine-tuning: start from the pretrained encoder weights ---
    finetune_model = DecoderTransformer(config, n_signals_for_loss=len(SIGNALS_FOR_LOSS))
    finetune_model.load_state_dict(model.state_dict(), strict=False)
    optimizer_full = torch.optim.AdamW(finetune_model.parameters(), lr=config.finetune_lr)

    finetune_model, ft_train_arr, ft_test_arr = finetune(
        finetune_model, optimizer_full, finetune_batcher, config.batch_size, pos_weight,
        epochs=config.finetune_epochs, show_every=200 if verbose else config.finetune_epochs + 1, verbose=verbose,
    )

    final_test_loss, final_test_auroc, final_test_ap = estimate_loss_bolus_head(
        finetune_model, finetune_batcher, config.batch_size, "test", pos_weight
    )
    final_train_loss, final_train_auroc, final_train_ap = estimate_loss_bolus_head(
        finetune_model, finetune_batcher, config.batch_size, "train", pos_weight
    )

    result = {
        "held_out": held_out_subject,
        "persistence_mse": persistence_baseline,
        "pretrain_test_mse": pretrain_test_loss,
        "train_auroc": final_train_auroc, "train_ap": final_train_ap,
        "test_auroc": final_test_auroc, "test_ap": final_test_ap,
        "finetune_test_loss": final_test_loss, "finetune_train_loss": final_train_loss,
        "pretrain_test_loss": pretrain_test_loss, "pretrain_train_loss": pretrain_train_loss,
    }
    if verbose:
        print(f"\n\nFOLD RESULT (held out {held_out_subject}): \n"
              f"pretrain_test_mse={pretrain_test_loss:.4f} (persistence={persistence_baseline:.4f}), "
              f"test_auroc={final_test_auroc:.4f}, test_ap={final_test_ap:.4f},\n"
              f"finetune_test_loss={final_test_loss:.4f}, final_train_loss={final_train_loss:.4f}\n"
              f"pretrain_test_loss={pretrain_test_loss:.4f}, pretrain_train_loss={pretrain_train_loss:.4f}")

    histories = {
        "pretrain_train": pt_train_arr, "pretrain_test": pt_test_arr,
        "finetune_train": ft_train_arr, "finetune_test": ft_test_arr,
    }
    return result, histories


def run_loso_sweep(all_subjects: List[int], config: Config = Config(),
                    data_dir: str = "bolus_data", verbose: bool = True):
    """Runs one LOSO fold per subject. Returns (results_df, last_fold_histories)."""
    loso_results = []
    last_fold_histories = None

    for held_out in all_subjects:
        result, histories = run_fold(held_out, all_subjects, config, data_dir, verbose)
        loso_results.append(result)
        last_fold_histories = histories

    results_df = pd.DataFrame(loso_results)
    return results_df, last_fold_histories
