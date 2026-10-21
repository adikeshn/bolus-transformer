from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import LOG_SIGNALS, NORMALIZED_SIGNALS, SIGNALS, SIGNALS_FOR_LOSS, Config
from .data_loading import load_data
from .features import add_bolus_features, add_time_features, make_labels


@dataclass(frozen=True)
class Sequence:
    key: str
    features: pd.DataFrame
    mask: np.ndarray
    labels: np.ndarray
    minutes: np.ndarray
    bolus_minutes: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def build_features(df: pd.DataFrame, bolus_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = add_bolus_features(df, bolus_df, config.iob_peak, config.iob_dia)
    df = add_time_features(df)
    for col in LOG_SIGNALS:
        df[col] = np.log1p(df[col])
    return df[SIGNALS]


@lru_cache(maxsize=None)
def load_sequence(subject: int, split: str, config: Config, data_dir: str) -> Sequence:
    df, bolus_df = load_data(subject, split, data_dir)
    features = build_features(df, bolus_df, config)
    minutes = df["minutes"].to_numpy()
    bolus_minutes = bolus_df["minutes"].to_numpy()
    return Sequence(
        key=f"{subject}-{split}",
        features=features,
        mask=features[SIGNALS_FOR_LOSS].notna().to_numpy(dtype=np.float32),
        labels=make_labels(minutes, bolus_minutes, config.label_window_minutes),
        minutes=minutes,
        bolus_minutes=bolus_minutes,
    )


def fit_norm_stats(seqs: List[Sequence]) -> Tuple[pd.Series, pd.Series]:
    pooled = pd.concat([s.features[NORMALIZED_SIGNALS] for s in seqs])
    means, stds = pooled.mean(), pooled.std()
    return means, stds.where(stds > 1e-8, 1.0)


def normalize(seq: Sequence, stats: Tuple[pd.Series, pd.Series]) -> Dict[str, np.ndarray]:
    means, stds = stats
    df = seq.features.copy()
    df[NORMALIZED_SIGNALS] = (df[NORMALIZED_SIGNALS] - means) / stds
    return {
        "x": df[SIGNALS].to_numpy(dtype=np.float32),
        "y": df[SIGNALS_FOR_LOSS].fillna(0.0).to_numpy(dtype=np.float32),
        "mask": seq.mask,
        "labels": seq.labels,
    }
