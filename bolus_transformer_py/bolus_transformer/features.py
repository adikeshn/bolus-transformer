"""Insulin-on-board decay, carb windowing, and cyclic time-of-day features."""
import numpy as np
import pandas as pd


def iob_falling(t, peak: float, dia: float):
    """Fraction of a bolus still active `t` minutes after dosing, on the falling
    (post-peak) side of the decay curve. See `_iob_fraction_matrix` for the full
    piecewise curve used elsewhere in the pipeline."""
    return 1 - peak / dia - (2 / (dia * (dia - peak))) * ((dia * t - t**2 / 2) - (dia * peak - peak**2 / 2))


def generate_cyclical_time_encoding(inp_df: pd.DataFrame) -> pd.DataFrame:
    """Adds sin/cos encodings of the 5-minute slot within the day (288 slots/day)."""
    inp_df["slot"] = (inp_df["date_time"].dt.hour * 60 + inp_df["date_time"].dt.minute) // 5
    inp_df["sin_t"] = np.sin(2 * np.pi * inp_df["slot"] / 288)
    inp_df["cos_t"] = np.cos(2 * np.pi * inp_df["slot"] / 288)
    return inp_df


def _iob_fraction_matrix(t: np.ndarray, peak: float, dia: float) -> np.ndarray:
    """Piecewise insulin-on-board decay curve: rises to 1.0 at `peak` minutes, then
    decays to 0 by `dia` minutes (duration of insulin action)."""
    frac = np.zeros_like(t)
    rising = (t > 0) & (t <= peak)
    falling = (t > peak) & (t <= dia)
    frac[rising] = 1 - (t[rising] ** 2) / (dia * peak)
    frac[falling] = (
        1 - peak / dia
        - (2 / (dia * (dia - peak)))
        * ((dia * t[falling] - t[falling] ** 2 / 2) - (dia * peak - peak ** 2 / 2))
    )
    return frac


def calc_IOB(inp_df: pd.DataFrame, bolus: pd.DataFrame, peak: float, dia: float) -> pd.DataFrame:
    """Insulin-on-board and self-reported-carbs features, with the bolus dosed at
    t=0 counted as fully active. Used when building pretraining inputs."""
    row_minutes = inp_df["minutes"].to_numpy()
    bolus_minutes = bolus["running_minutes"].to_numpy()
    bolus_doses = bolus["dose"].to_numpy()
    bolus_carbs = bolus["bwz_carb_input"].to_numpy()

    t = row_minutes[:, None] - bolus_minutes[None, :]

    frac = _iob_fraction_matrix(t, peak, dia)
    frac[t == 0] = 1.0
    inp_df["running_IOB"] = (frac * bolus_doses[None, :]).sum(axis=1)

    carb_window = (t >= 0) & (t < 5)
    inp_df["self_reported_carbs"] = np.where(carb_window, bolus_carbs[None, :], 0).sum(axis=1)
    return inp_df


def calc_IOB_no_leak(inp_df: pd.DataFrame, bolus: pd.DataFrame, peak: float, dia: float) -> pd.DataFrame:
    """Same as `calc_IOB`, but excludes t=0 so a bolus can't leak into its own
    fine-tuning label window. Used when building fine-tuning inputs."""
    row_minutes = inp_df["minutes"].to_numpy()
    bolus_minutes = bolus["running_minutes"].to_numpy()
    bolus_doses = bolus["dose"].to_numpy()
    bolus_carbs = bolus["bwz_carb_input"].to_numpy()

    t = row_minutes[:, None] - bolus_minutes[None, :]

    frac = _iob_fraction_matrix(t, peak, dia)
    inp_df["running_IOB"] = (frac * bolus_doses[None, :]).sum(axis=1)

    carb_window = (t > 0) & (t < 5)  # t == 0 excluded
    inp_df["self_reported_carbs"] = np.where(carb_window, bolus_carbs[None, :], 0).sum(axis=1)
    return inp_df
