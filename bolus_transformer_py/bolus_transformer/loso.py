import copy
from typing import Dict, List, Sequence as Seq

import numpy as np

from .baselines import run_baselines
from .config import Config
from .metrics import alarm_metrics, choose_threshold, ranking_metrics
from .model import CausalTransformer
from .preprocessing import fit_norm_stats, load_sequence, normalize
from .training import (DEVICE, bolus_probs, finetune, pooled_forecast_mse, pooled_persistence_mse,
                       pretrain, set_seed)

METHODS = ("transformer", "transformer_nopretrain", "gbm", "time_of_day")


def _recordings(seqs, probs_list, config: Config) -> List[dict]:
    off = config.eval_offset
    return [{"probs": p, "minutes": s.minutes[off:], "bolus_minutes": s.bolus_minutes}
            for s, p in zip(seqs, probs_list)]


def _score(method, val_seqs, val_probs, test_seqs, test_probs, config: Config) -> dict:
    off = config.eval_offset
    labels = np.concatenate([s.labels[off:] for s in test_seqs])
    probs = np.concatenate(test_probs)
    auroc, ap = ranking_metrics(probs, labels)
    window = config.label_window_minutes
    thr = choose_threshold(_recordings(val_seqs, val_probs, config), window, config.target_false_alarms_per_day)
    return {"method": method, "test_auroc": auroc, "test_ap": ap, "base_rate": float(labels.mean()),
            **alarm_metrics(_recordings(test_seqs, test_probs, config), window, thr)}


def run_fold(test_subject: int, val_subject: int, train_subjects: Seq[int], config: Config, seed: int,
             data_dir: str, test_splits: Seq[str] = ("training", "testing"),
             methods: Seq[str] = METHODS, verbose: bool = True) -> List[dict]:
    if verbose:
        print(f"\n=== test {test_subject} | val {val_subject} | train {list(train_subjects)} | seed {seed} ===")
    set_seed(seed)

    train_seqs = [load_sequence(s, "training", config, data_dir) for s in train_subjects]
    val_seqs = [load_sequence(val_subject, "training", config, data_dir)]
    test_seqs = [load_sequence(test_subject, sp, config, data_dir) for sp in test_splits]

    stats = fit_norm_stats(train_seqs)
    train_arr = {s.key: normalize(s, stats) for s in train_seqs}
    val_arr = [normalize(s, stats) for s in val_seqs]
    test_arr = [normalize(s, stats) for s in test_seqs]

    probs: Dict[str, Dict[str, List[np.ndarray]]] = {}
    extra: Dict[str, dict] = {}

    variants = []
    if "transformer" in methods:
        model, _ = pretrain(CausalTransformer(config).to(DEVICE), train_arr, val_arr, config, verbose)
        extra["transformer"] = {
            "pretrain_test_mse": pooled_forecast_mse(model, test_arr, config),
            "persistence_test_mse": pooled_persistence_mse(test_arr, config),
        }
        variants.append(("transformer", copy.deepcopy(model)))
    if "transformer_nopretrain" in methods:
        variants.append(("transformer_nopretrain", CausalTransformer(config).to(DEVICE)))
    for name, m in variants:
        m, _ = finetune(m, train_arr, val_arr, config, verbose)
        probs[name] = {"val": [bolus_probs(m, a, config) for a in val_arr],
                       "test": [bolus_probs(m, a, config) for a in test_arr]}

    probs.update(run_baselines(train_seqs, {"val": val_seqs, "test": test_seqs}, config, seed, methods))

    results = []
    for method in methods:
        r = _score(method, val_seqs, probs[method]["val"], test_seqs, probs[method]["test"], config)
        r.update({"test_subject": test_subject, "val_subject": val_subject, "seed": seed, **extra.get(method, {})})
        results.append(r)
        if verbose:
            print(f"  {method:24s} AUROC {r['test_auroc']:.3f}  AP {r['test_ap']:.3f}  "
                  f"recall@{config.target_false_alarms_per_day:g}FA/day {r['recall_at_fa']:.2f} "
                  f"(actual {r['fa_per_day']:.1f} FA/day)")
    return results


def run_loso(subjects: Seq[int], config: Config, seeds: Seq[int], data_dir: str,
             test_splits: Seq[str] = ("training", "testing"), methods: Seq[str] = METHODS,
             verbose: bool = True) -> List[dict]:
    subjects = list(subjects)
    results = []
    for i, test_subject in enumerate(subjects):
        val_subject = subjects[(i + 1) % len(subjects)]
        train_subjects = [s for s in subjects if s not in (test_subject, val_subject)]
        for seed in seeds:
            results += run_fold(test_subject, val_subject, train_subjects, config, seed,
                                data_dir, test_splits, methods, verbose)
    return results
