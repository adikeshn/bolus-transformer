from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .config import Config
from .preprocessing import Sequence

TOD_FEATURES = ["sin_t", "cos_t"]


def tabular_features(seq: Sequence) -> pd.DataFrame:
    f = seq.features
    g = f["glucose_level"]
    b = np.sort(seq.bolus_minutes)
    last = np.searchsorted(b, seq.minutes, side="right") - 1
    since = np.where(last >= 0, seq.minutes - b[np.clip(last, 0, None)], np.nan)
    return pd.DataFrame({
        "glucose": g,
        "d_glucose_15": g - g.shift(3),
        "d_glucose_30": g - g.shift(6),
        "d_glucose_60": g - g.shift(12),
        "iob": f["running_IOB"],
        "carbs_3h": np.expm1(f["self_reported_carbs"]).rolling(36, min_periods=1).sum(),
        "min_since_bolus": since,
        "heart_rate": f["basis_heart_rate"],
        "sin_t": f["sin_t"],
        "cos_t": f["cos_t"],
    })


def run_baselines(train: List[Sequence], eval_seqs: Dict[str, List[Sequence]],
                  config: Config, seed: int, methods) -> Dict[str, Dict[str, List[np.ndarray]]]:
    off = config.eval_offset
    X_train = pd.concat([tabular_features(s).iloc[off:] for s in train])
    y_train = np.concatenate([s.labels[off:] for s in train])
    eval_X = {k: [tabular_features(s).iloc[off:] for s in seqs] for k, seqs in eval_seqs.items()}

    models = {}
    if "time_of_day" in methods:
        models["time_of_day"] = (
            LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train[TOD_FEATURES], y_train),
            TOD_FEATURES,
        )
    if "gbm" in methods:
        models["gbm"] = (
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed,
            ).fit(X_train, y_train),
            list(X_train.columns),
        )

    return {
        name: {k: [clf.predict_proba(X[cols])[:, 1] for X in Xs] for k, Xs in eval_X.items()}
        for name, (clf, cols) in models.items()
    }
