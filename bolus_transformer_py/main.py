"""CLI entry point: runs the full LOSO cross-validation sweep and reports/plots results.

Usage:
    python main.py --data-dir bolus_data --subjects 563 559 570 575 588 591
"""
import argparse

import matplotlib.pyplot as plt

from bolus_transformer.config import Config
from bolus_transformer.loso import run_loso_sweep

# 2018 cohort (Basis sensor band) -- the six subjects used in the reported LOSO results.
DEFAULT_SUBJECTS = [563, 559, 570, 575, 588, 591]


def parse_args():
    parser = argparse.ArgumentParser(description="Run LOSO cross-validation for the bolus-timing transformer.")
    parser.add_argument("--data-dir", default="bolus_data", help="Directory containing the OhioT1DM XML files.")
    parser.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS,
                         help="Subject IDs to include in the LOSO sweep.")
    parser.add_argument("--plot-out", default=None,
                         help="If set, save the last fold's training curves here instead of showing them.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step training logs.")
    return parser.parse_args()


def plot_histories(histories: dict, out_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(histories["pretrain_train"]); axes[0, 0].set_title("pretrain train loss")
    axes[0, 1].plot(histories["pretrain_test"]); axes[0, 1].set_title("pretrain test loss")
    axes[1, 0].plot(histories["finetune_train"]); axes[1, 0].set_title("finetune train loss")
    axes[1, 1].plot(histories["finetune_test"]); axes[1, 1].set_title("finetune test loss")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Saved training curves to {out_path}")
    else:
        plt.show()


def main():
    args = parse_args()
    config = Config()

    results_df, last_fold_histories = run_loso_sweep(
        args.subjects, config=config, data_dir=args.data_dir, verbose=not args.quiet
    )

    print()
    print(results_df)
    print("Means:\n")
    print(results_df[[
        "pretrain_test_mse", "test_auroc", "test_ap",
        "pretrain_train_loss", "pretrain_test_loss",
        "finetune_train_loss", "finetune_test_loss",
    ]].agg(["mean", "std"]))

    if last_fold_histories is not None:
        plot_histories(last_fold_histories, args.plot_out)


if __name__ == "__main__":
    main()
