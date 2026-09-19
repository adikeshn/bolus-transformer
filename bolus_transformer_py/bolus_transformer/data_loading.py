"""Reads OhioT1DM XML exports and aligns signals onto the 5-minute glucose grid."""
import pandas as pd

from .config import START_TIME, XML_SIGNALS


def get_bolus_df(subject_num: int, split: str, data_dir: str = "bolus_data") -> pd.DataFrame:
    """Load and time-index the logged bolus events for one subject/split."""
    bolus_df = pd.read_xml(f"{data_dir}/{subject_num}-ws-{split}.xml", xpath="./bolus/event")
    bolus_df["date_time"] = pd.to_datetime(bolus_df["ts_begin"], format="%d-%m-%Y %H:%M:%S")
    bolus_df["running_minutes"] = (bolus_df["date_time"] - START_TIME).dt.total_seconds() / 60
    return bolus_df.sort_values("running_minutes").reset_index(drop=True)


def load_data(subject_num: int, split: str, data_dir: str = "bolus_data"):
    """Load each physiological signal, align them onto the glucose (5-min) grid via
    a backward as-of merge, and trim to the window where every wearable signal has
    coverage.

    Returns (df, bolus_df): the aligned signal frame and the subject's bolus log.
    """
    bolus_df = get_bolus_df(subject_num, split, data_dir)

    signal_dfs = {}
    for signal in XML_SIGNALS:
        sig_df = pd.read_xml(f"{data_dir}/{subject_num}-ws-{split}.xml", xpath=f"./{signal}/event")
        sig_df["date_time"] = pd.to_datetime(sig_df["ts"], format="%d-%m-%Y %H:%M:%S")
        sig_df = sig_df[["date_time", "value"]].rename(columns={"value": signal})
        signal_dfs[signal] = sig_df.sort_values("date_time").reset_index(drop=True)

    df = signal_dfs["glucose_level"]
    for signal in XML_SIGNALS[1:]:
        df = pd.merge_asof(
            df, signal_dfs[signal],
            on="date_time",
            direction="backward",
            tolerance=pd.Timedelta("2.5min"),
        )

    df["minutes"] = (df["date_time"] - START_TIME).dt.total_seconds() / 60
    df["running_IOB"] = 0.0
    df["self_reported_carbs"] = 0.0

    wearable_cols = ["basis_heart_rate", "basis_gsr", "basis_skin_temperature", "basis_air_temperature"]
    overlap_start = max(df.loc[df[col].notna(), "date_time"].min() for col in wearable_cols)
    overlap_end = min(df.loc[df[col].notna(), "date_time"].max() for col in wearable_cols)
    df = df[(df["date_time"] >= overlap_start) & (df["date_time"] <= overlap_end)].reset_index(drop=True)
    return df, bolus_df
