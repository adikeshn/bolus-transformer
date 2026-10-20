import numpy as np
import pandas as pd


def iob_fraction(t: np.ndarray, peak: float, dia: float) -> np.ndarray:
    frac = np.zeros_like(t, dtype=np.float64)
    frac[t == 0] = 1.0
    rising = (t > 0) & (t <= peak)
    falling = (t > peak) & (t <= dia)
    frac[rising] = 1 - (t[rising] ** 2) / (dia * peak)
    tf = t[falling]
    frac[falling] = 1 - peak / dia - (2 / (dia * (dia - peak))) * (
        (dia * tf - tf ** 2 / 2) - (dia * peak - peak ** 2 / 2)
    )
    return frac


def add_bolus_features(df: pd.DataFrame, bolus_df: pd.DataFrame, peak: float, dia: float) -> pd.DataFrame:
    t = df["minutes"].to_numpy()[:, None] - bolus_df["minutes"].to_numpy()[None, :]
    out = df.copy()
    out["running_IOB"] = iob_fraction(t, peak, dia) @ bolus_df["dose"].to_numpy(dtype=np.float64)
    just_logged = (t >= 0) & (t < 5)
    out["self_reported_carbs"] = just_logged @ bolus_df["bwz_carb_input"].to_numpy(dtype=np.float64)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    slot = (out["date_time"].dt.hour * 60 + out["date_time"].dt.minute) // 5
    out["sin_t"] = np.sin(2 * np.pi * slot / 288)
    out["cos_t"] = np.cos(2 * np.pi * slot / 288)
    return out


def make_labels(row_minutes: np.ndarray, bolus_minutes: np.ndarray, window: float) -> np.ndarray:
    b = np.sort(bolus_minutes)
    after_t = np.searchsorted(b, row_minutes, side="right")
    by_end = np.searchsorted(b, row_minutes + window, side="right")
    return (by_end > after_t).astype(np.float32)
