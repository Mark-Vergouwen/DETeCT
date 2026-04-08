import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

def load_tsf_features(feature_prefix="m_", time_col = "moy", path = "", DATA_TYPE = "public", label_map = {}):
    
    """
    Load and prepare time-series feature data 

    Returns:
        features_df (pd.DataFrame): merged feature + weather dataframe
    """
    
    # 1. Load public data
    tsf_files = list(path.glob(f"{feature_prefix}*"))
    features_df = pd.concat((pd.read_csv(f) for f in tsf_files), ignore_index=True)
    features_df = features_df.rename(columns={"unique_id": "ean_id"})

    sort_cols = ["ean_id"] + ([time_col] if time_col is not None else [])
    features_df = features_df.sort_values(sort_cols)
    print(f"Loaded {len(tsf_files)} files into one DataFrame with shape {features_df.shape}")
    
    features_df["label"] = None
    
    if label_map is not None:
        for label, eans in label_map.items():
            features_df.loc[features_df["ean_id"].isin(eans), "label"] = label
            
    return features_df
