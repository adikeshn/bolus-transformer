import pandas as pd

from .config import START_TIME, WEARABLE_SIGNALS, XML_SIGNALS

TS_FORMAT = "%d-%m-%Y %H:%M:%S"


def _xml_path(subject_num: int, split: str, data_dir: str) -> str:
    return f"{data_dir}/{subject_num}-ws-{split}.xml"


def to_minutes(ts: pd.Series) -> pd.Series:
    return (ts - START_TIME).dt.total_seconds() / 60


def get_bolus_df(subject_num: int, split: str, data_dir: str = "bolus_data") -> pd.DataFrame:
    bolus_df = pd.read_xml(_xml_path(subject_num, split, data_dir), xpath="./bolus/event")
    bolus_df["date_time"] = pd.to_datetime(bolus_df["ts_begin"], format=TS_FORMAT)
    bolus_df["minutes"] = to_minutes(bolus_df["date_time"])
    if "bwz_carb_input" not in bolus_df:
        bolus_df["bwz_carb_input"] = 0.0
    bolus_df["bwz_carb_input"] = bolus_df["bwz_carb_input"].fillna(0.0)
    return bolus_df.sort_values("minutes").reset_index(drop=True)


def load_data(subject_num: int, split: str, data_dir: str = "bolus_data"):
    bolus_df = get_bolus_df(subject_num, split, data_dir)

    signal_dfs = {}
    for signal in XML_SIGNALS:
        sig_df = pd.read_xml(_xml_path(subject_num, split, data_dir), xpath=f"./{signal}/event")
        sig_df["date_time"] = pd.to_datetime(sig_df["ts"], format=TS_FORMAT)
        sig_df = sig_df[["date_time", "value"]].rename(columns={"value": signal})
        signal_dfs[signal] = sig_df.sort_values("date_time").reset_index(drop=True)

    df = signal_dfs["glucose_level"]
    for signal in WEARABLE_SIGNALS:
        df = pd.merge_asof(
            df, signal_dfs[signal],
            on="date_time", direction="backward", tolerance=pd.Timedelta("2.5min"),
        )

    overlap_start = max(df.loc[df[col].notna(), "date_time"].min() for col in WEARABLE_SIGNALS)
    overlap_end = min(df.loc[df[col].notna(), "date_time"].max() for col in WEARABLE_SIGNALS)
    df = df[(df["date_time"] >= overlap_start) & (df["date_time"] <= overlap_end)].reset_index(drop=True)
    df["minutes"] = to_minutes(df["date_time"])
    return df, bolus_df
