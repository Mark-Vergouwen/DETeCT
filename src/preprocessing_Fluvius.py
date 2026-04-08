# src/preprocessing_Fluvius.py
import pandas as pd
from typing import Optional, Dict


def preprocess_electricity_data(
    df: pd.DataFrame,
    *,
    ean_col: str,
    datetime_col: str,
    afname_col: str,
    afname_unit: str,                # "kW" or "kWh"
    interval_minutes: int,
    injectie_col: Optional[str] = None,
    productie_col: Optional[str] = None,
    labelled_data: bool = False,
    label_cols: Optional[Dict[str, str]] = None,  # {"HP": "...", "EV": "...", "PV": "..."}
) -> pd.DataFrame:
    """
    General preprocessing for electricity consumption data.

    Parameters
    ----------
    ean_col : str
        Column name of EAN identifier (integer).
    datetime_col : str
        Column name of datetime variable.
    afname_col : str
        Column name of consumption variable.
    afname_unit : {"kW", "kWh"}
        Unit of the consumption variable.
    interval_minutes : int
        Length of measurement interval (e.g. 15, 30, 60).
    injectie_col : str, optional
        Column name of injection variable.
    productie_col : str, optional
        Column name of production variable.
    labelled_data : bool
        Whether HP / EV / PV labels are present.
    label_cols : dict, required if labelled_data=True
        Mapping {"HP": colname, "EV": colname, "PV": colname}

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with kWh variables and standard time features.
    """

    df = df.copy()

    # ---------------------
    # EAN + datetime
    # ---------------------
    df["EAN_ID"] = df[ean_col].astype("int64")
    df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")

    # ---------------------
    # Time features (always recompute)
    # ---------------------
    df["year"] = df["datetime"].dt.year
    df["moy"] = df["datetime"].dt.month
    df["week"] = df["datetime"].dt.isocalendar().week.astype("int16")
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["min"] = df["datetime"].dt.minute

    # ---------------------
    # Helper: convert to kWh
    # ---------------------
    def to_kwh(series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        if afname_unit.lower() == "kw":
            return series * (interval_minutes / 60)
        elif afname_unit.lower() == "kwh":
            return series
        else:
            raise ValueError("afname_unit must be 'kW' or 'kWh'")

    # ---------------------
    # Consumption / injection / production
    # ---------------------
    df["afname_kwh"] = to_kwh(df[afname_col])

    if injectie_col is not None and injectie_col in df.columns:
        df["injectie_kwh"] = to_kwh(df[injectie_col])

    # ---------------------
    # Labels (optional)
    # ---------------------
    if labelled_data:
        if label_cols is None:
            raise ValueError("label_cols must be provided when labelled_data=True")

        for label, col in label_cols.items():
            if col not in df.columns:
                raise ValueError(f"Label column '{col}' not found in dataframe")
            df[label] = df[col].astype("int8")

    # ---------------------
    # Final column selection (conservative)
    # ---------------------
    keep_cols = [
        "EAN_ID", "datetime",
        "year", "moy", "week", "day", "hour", "min",
        "afname_kwh",
    ]

    for col in ["injectie_kwh", "HP", "EV", "PV"]:
        if col in df.columns:
            keep_cols.append(col)

    return df[keep_cols]


