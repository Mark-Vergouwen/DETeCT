import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

def load_sma_features(weather_prefix="m_w_weather",
                      feature_prefix="m_w_",
                      time_col="moy", 
                      path = "",
                      DATA_TYPE = "public",
                      label_map = None):
    """
    Load and merge SmartMeterAnalytics features with weather data.

    Returns:
        all_df (pd.DataFrame): merged feature + weather dataframe
    """
    # Load features
        
    # 1. load metering features
    feature_files = [
        f for f in path.glob(f"{feature_prefix}*")
        if not f.name.startswith(weather_prefix)
    ]
        
    features_df = pd.concat((pd.read_csv(f) for f in feature_files), ignore_index=True)
    features_df = features_df.sort_values('unique_id')

    # 2. Load weather features
    weather_files = list(path.glob(f"{weather_prefix}*"))

    weather_df = pd.concat((pd.read_csv(f) for f in weather_files), ignore_index=True)
    weather_df = weather_df.sort_values('unique_id')

    # 3. Merging
    cor_cols = weather_df.filter(regex=r'^cor_').columns
    if time_col is not None:
        weather_df = weather_df[['unique_id', time_col, *cor_cols]]
        merge_cols = ['unique_id', time_col]
    else:
        weather_df = weather_df[['unique_id', *cor_cols]]
        merge_cols = ['unique_id']
        
    all_df = features_df.merge(weather_df, on=merge_cols, how='inner')
        
    # 4. Assign labels (works for private, public 2022 and public 2024)
    all_df = all_df.rename(columns={"unique_id": "ean_id"})
    all_df["label"] = None
    
    if label_map is not None:
        for label, eans in label_map.items():
            all_df.loc[all_df["ean_id"].isin(eans), "label"] = label
    
    return all_df