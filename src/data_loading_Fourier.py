import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np


def load_Fourier_features(feature_prefix="merged_features_collm.csv", time_col = "moy", path = "", DATA_TYPE = "public", label_map = {}):
    """
    Load and prepare time-series feature data 

    Returns:
        features_df (pd.DataFrame): merged feature + weather dataframe
    """
    # 1. Load all monthly tsfeature CSVs
    fourier_files = list(path.glob(f"{feature_prefix}*"))

    features_df = pd.concat((pd.read_csv(f) for f in fourier_files), ignore_index=True)
    cols_to_drop = [c for c in features_df.columns if c != 'moy' and c != 'year' and c.startswith(('year_', 'moy_'))]
    features_df = features_df.drop(columns=cols_to_drop)
    
    sort_cols = ["EAN_ID"] + ([time_col] if time_col is not None else [])
    features_df = features_df.sort_values(sort_cols)
    print(f"Loaded {len(fourier_files)} files into one DataFrame with shape {features_df.shape}")

    features_df = features_df.rename(columns={"EAN_ID": "ean_id"})

    if label_map is not None:
        for label, eans in label_map.items():
            features_df.loc[features_df["ean_id"].isin(eans), "label"] = label
             
    
    return features_df