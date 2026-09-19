"""Per-signal normalization and assembly of model-ready feature matrices."""
import numpy as np
import pandas as pd

from .config import SIGNALS, SIGNALS_FOR_LOSS
from .data_loading import load_data
from .features import calc_IOB, calc_IOB_no_leak, generate_cyclical_time_encoding


def preprocess_data(df: pd.DataFrame, signals, means: dict, stds: dict):
    """Log-transform GSR and carbs, z-score the raw signals, and fill remaining NaNs.

    Returns (df, df[SIGNALS_FOR_LOSS]) — the full processed frame and the subset of
    columns used as the pretraining regression target.
    """
    df["basis_gsr"] = np.log1p(df["basis_gsr"])
    df["self_reported_carbs"] = np.log1p(df["self_reported_carbs"])
    for col in signals[:-2]:  # skip sin_t/cos_t, which are already in [-1, 1]
        df[col] = (df[col] - means[col]) / stds[col]
    df = df.fillna(0.0)
    return df, df[SIGNALS_FOR_LOSS]


def get_data(subject_num: int, split: str, iob_peak: float, iob_dia: float, data_dir: str = "bolus_data"):
    """Load raw signals for pretraining, with same-timestep IOB leakage allowed.

    Returns (df, bolus_df[["dose", "date_time"]], mask).
    """
    df, bolus_df = load_data(subject_num, split, data_dir)
    df = calc_IOB(df, bolus_df, iob_peak, iob_dia)
    df_p = generate_cyclical_time_encoding(df)
    mask = (~df_p[SIGNALS_FOR_LOSS].isna()).to_numpy().astype(np.float32)
    return df_p, bolus_df[["dose", "date_time"]], mask


def build_leak_free_x(subject_num: int, split: str, means: dict, stds: dict,
                       iob_peak: float, iob_dia: float, data_dir: str = "bolus_data"):
    """Load and normalize signals for fine-tuning, with leak-free IOB/carb features
    (the bolus at t=0 is excluded so it can't trivially predict its own label)."""
    df, bolus_df = load_data(subject_num, split, data_dir)
    df = calc_IOB_no_leak(df, bolus_df, iob_peak, iob_dia)
    df = generate_cyclical_time_encoding(df)
    df["basis_gsr"] = np.log1p(df["basis_gsr"])
    df["self_reported_carbs"] = np.log1p(df["self_reported_carbs"])
    for col in SIGNALS[:-2]:
        df[col] = (df[col] - means[col]) / stds[col]
    df = df.fillna(0.0)
    return df, bolus_df


def gen_finetuning_data(subject_ids: list, subject_ids_test: list, means: dict, stds: dict,
                         iob_peak: float, iob_dia: float, threshold: int, data_dir: str = "bolus_data"):
    """Builds per-subject (features, bolus_label) arrays for fine-tuning. Each
    timestep within `threshold` tokens *before* a logged bolus is labeled positive.

    Returns (bolus_train, bolus_test): dicts mapping subject_id -> (features, labels).
    """
    bolus_train, bolus_test = {}, {}

    for subject in subject_ids + subject_ids_test:
        df, bolus_df = build_leak_free_x(subject, "training", means, stds, iob_peak, iob_dia, data_dir)
        df = pd.merge_asof(
            df, bolus_df[["date_time", "dose"]],
            on="date_time",
            direction="backward",
            tolerance=pd.Timedelta("5min"),
        )
        df["dose"] = df["dose"].notnull().astype(int)
        df["dose"] = (
            df["dose"][::-1].rolling(window=threshold, min_periods=1).max()[::-1].fillna(0).astype(int)
        )
        bolus_labels = df["dose"].to_numpy().astype(np.float32)
        features = df[SIGNALS].to_numpy().astype(np.float32)

        if subject in subject_ids:
            bolus_train[subject] = (features, bolus_labels)
        else:
            bolus_test[subject] = (features, bolus_labels)

    return bolus_train, bolus_test
