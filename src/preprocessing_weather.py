# src/preprocessing_weather.py
import pandas as pd

def preprocess_weather(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Preprocess weather data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw weather dataframe with at least the following columns:
        - 'datetime' (string, datetime-like)
        - 'T', 'U', 'Ff', 'DD', 'N', 'GHI', 'DNI'  (temperature, humidity, mean wind speed, wind direction, cloud coverage, irradiation ; continuous variables)

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with:
        - datetime as pandas datetime
        - new columns: 'year', 'moy', 'day', 'hour', 'min'
        - continuous variables as float
        - categorical variables as category dtype
    """

    # --- datetime handling ---
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M", errors="coerce")
    df["year"] = df["datetime"].dt.year
    df["moy"] = df["datetime"].dt.month      # month of year
    df["day"] = df["datetime"].dt.day        # day of month
    df["hour"] = df["datetime"].dt.hour

    # --- continuous variables ---
    continuous_vars = ["T", "U", "Ff", 'DD', 'N', 'GHI', 'DNI']
    for col in continuous_vars:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df.drop(columns=["datetime"], inplace=True)

    return df

