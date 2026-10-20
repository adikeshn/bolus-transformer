import argparse
import dataclasses
import os
from typing import List, Optional

import pandas as pd

from bolus_transformer.config import Config
from bolus_transformer.loso import METHODS, run_loso

DEFAULT_SUBJECTS = [559, 563, 570, 575, 588, 591]
SUMMARY_COLS = ["test_auroc", "test_ap", "recall_at_fa", "fa_per_day", "median_lead_min"]


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Leave-one-subject-out evaluation for bolus timing.")
    p.add_argument("--data-dir", default="bolus_data")
    p.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--test-splits", nargs="+", default=["training", "testing"])
    p.add_argument("--out-dir", default="results")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def quick_config(config: Config) -> Config:
    return dataclasses.replace(config, pretrain_steps=60, finetune_steps=60, eval_every=20, patience=3)


def summarize(df: pd.DataFrame, config: Config) -> str:
    fa = f"{config.target_false_alarms_per_day:g}"
    headers = ["Method", "AUROC", "AP", f"Bolus recall @ ≤{fa} FA/day", "Actual FA/day", "Median lead (min)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for method, g in df.groupby("method", sort=False):
        cells = [f"{g[c].mean():.3f} ± {g[c].std(ddof=0):.3f}" if c != "median_lead_min"
                 else f"{g[c].mean():.0f} ± {g[c].std(ddof=0):.0f}" for c in SUMMARY_COLS]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")

    out = "\n".join(lines)
    out += f"\n\nBase rate (positive rows): {df['base_rate'].mean():.3f}. "
    out += f"Runs per method: {df.groupby('method').size().iloc[0]} (folds × seeds).\n"
    if "pretrain_test_mse" in df:
        t = df.dropna(subset=["pretrain_test_mse"])
        if len(t):
            gain = 1 - t["pretrain_test_mse"] / t["persistence_test_mse"]
            out += (f"\nPretraining (30-min forecast, held-out subject): weighted MSE "
                    f"{t['pretrain_test_mse'].mean():.4f} vs persistence {t['persistence_test_mse'].mean():.4f} "
                    f"({gain.mean():.1%} ± {gain.std(ddof=0):.1%} lower).\n")
    return out


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    config = quick_config(Config()) if args.quick else Config()

    results = run_loso(args.subjects, config, args.seeds, args.data_dir,
                       args.test_splits, args.methods, verbose=not args.quiet)

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    summary = summarize(df, config)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary)
    return df


if __name__ == "__main__":
    main()
