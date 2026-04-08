import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

def load_ifeel_features(feature_prefix = "m_", time_col = "moy", path = " ", DATA_TYPE = "public", label_map = {}):
    """
    Load and prepare IFEEL monthly features (_collm.csv files).
    Combines global and peak-period features per household, drops unnecessary columns, and assigns labels.

    Returns:
        pd.DataFrame: Combined dataframe with features and `label` column.
    """
    # 1. Load all CSV files
    ifeel_files = list(path.glob(f"{feature_prefix}*"))

    features_df = pd.concat((pd.read_csv(f) for f in ifeel_files), ignore_index=True)
    features_df = features_df.sort_values(['ean_id'])
    print(f"Loaded {len(ifeel_files)} files into one DataFrame with shape {features_df.shape}")

    # 2. Combine global and peak-period features per household/month
    group_cols = ['ean_id'] + ([time_col] if time_col is not None else [])
    features_df = features_df.groupby(group_cols, as_index=False).max(numeric_only=False)

    # 3. Drop unnecessary peak-period columns
    drop_cols = [
        "peak_duration_max", "peak_duration_mean", "peak_duration_median",
        "peak_duration_min", "peak_duration_std",
        "peak_time_max", "peak_time_mean", "peak_time_median",
        "peak_time_min", "peak_time_std"
    ]
    features_df = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])

    if label_map is not None: 
        for label, eans in label_map.items(): 
            features_df.loc[features_df["ean_id"].isin(eans), "label"] = label
           
    
    return features_df
