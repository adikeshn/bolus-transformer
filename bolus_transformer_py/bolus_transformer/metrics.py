from typing import List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

MINUTES_PER_ROW = 5.0


def ranking_metrics(probs: np.ndarray, labels: np.ndarray):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan"), float("nan")
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


def _alarm_times(probs, minutes, threshold) -> np.ndarray:
    above = probs >= threshold
    return minutes[above & ~np.concatenate([[False], above[:-1]])]


def _alarm_counts(probs, minutes, bolus_minutes, window, threshold):
    alarm_t = _alarm_times(probs, minutes, threshold)
    b = np.sort(bolus_minutes)
    hit = np.searchsorted(b, alarm_t + window, side="right") > np.searchsorted(b, alarm_t, side="right")
    return int((~hit).sum()), int(hit.sum())


def _evaluable_boluses(minutes, bolus_minutes, window) -> np.ndarray:
    lo = np.searchsorted(minutes, bolus_minutes - window, side="left")
    hi = np.searchsorted(minutes, bolus_minutes, side="left")
    return bolus_minutes[hi > lo]


def choose_threshold(recordings: List[dict], window: float, target_fa_per_day: float) -> float:
    pooled = np.concatenate([r["probs"] for r in recordings])
    days = len(pooled) * MINUTES_PER_ROW / 1440
    for thr in np.unique(np.quantile(pooled, np.linspace(0.5, 1.0, 501))):
        fa = sum(_alarm_counts(r["probs"], r["minutes"], r["bolus_minutes"], window, thr)[0] for r in recordings)
        if fa / days <= target_fa_per_day:
            return float(thr)
    return float(pooled.max()) + 1e-9


def alarm_metrics(recordings: List[dict], window: float, threshold: float) -> dict:
    n_rows = sum(len(r["probs"]) for r in recordings)
    days = n_rows * MINUTES_PER_ROW / 1440
    false_alarms, caught, leads, n_boluses = 0, 0, [], 0
    for r in recordings:
        false_alarms += _alarm_counts(r["probs"], r["minutes"], r["bolus_minutes"], window, threshold)[0]
        alarm_t = _alarm_times(r["probs"], r["minutes"], threshold)
        for b in _evaluable_boluses(r["minutes"], r["bolus_minutes"], window):
            n_boluses += 1
            before = alarm_t[(alarm_t >= b - window) & (alarm_t < b)]
            if len(before):
                caught += 1
                leads.append(b - before.min())
    return {
        "recall_at_fa": caught / n_boluses if n_boluses else float("nan"),
        "fa_per_day": false_alarms / days if days else float("nan"),
        "median_lead_min": float(np.median(leads)) if leads else float("nan"),
        "n_boluses": n_boluses,
        "threshold": threshold,
    }
