# src/merging.py

"""
Script to preprocess and merge Fluvius meter data with weather data.

Workflow:
1. Load and preprocess weather and pricing data.

2. Loop over all metering files:
    - Load and preprocess each raw metering file
    - Merge with weather data
    - Save intermediate results to 'Intermediate' folder
    
3. Call feature engineering functions
    - tsfeatures
    - SMA
    - IFEEL
    - Fourier
    
"""

#%% PRELIMINARIES

# ==========================================
# 1. Load packages and define paths
# ==========================================

import os
import yaml
from pathlib import Path
import glob
from tsfeatures import tsfeatures 
import pandas as pd
from src.preprocessing_weather import preprocess_weather
from src.preprocessing_prices import preprocess_prices
from src.preprocessing_Fluvius import preprocess_electricity_data

# --- 0. Load configuration ---

# Works whether you run the full script or just a block interactively.
try:
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = BASE_DIR.parent / "config.yml"
except NameError:
    # __file__ is not defined in interactive/block execution contexts
    BASE_DIR = Path.cwd()
    CONFIG_PATH = BASE_DIR / "config.yml"

with open("config.yml") as f:
    config = yaml.safe_load(f)
    
data_root = Path(config["data_root"])

# ==========================================
# 2. Configuration yearly/monthly/weekly
# ==========================================

def discover_subset_files(data_config):
    inter = Path(data_config["paths"]["data_intermediate"])
    prefix = data_config["schema"]["subsets"]

    return sorted(
        f for f in inter.glob("*.csv")
        if f.stem.startswith(prefix + "_")
    )

# For generalizability, choose aggregation level
DATA_TYPE = "private"  # Options: "private", "public",  "public2022" or "USER"
AGGREGATION_LEVEL = "yearly"

CONFIG = {
    "yearly": {
        "prefixes": {
            "sma_weather": "y_w_weather_2",
            "sma_features":  "y_w_2",
            "sma_weather_inj": "y_w_weather_inj",
            "sma_features_inj": "y_w_inj_",
            "tsf": "y_",
            "ifeel": "y_",
            "fourier": "y_"
        },
        "merge_on": ["ean_id"],  # Keys to merge datasets: only on ean_id
        "output_suffix": "_y",
        "time_col": None
    },
    "monthly": {
        "prefixes": {
            "sma_weather": "m_w_weather_2",
            "sma_features":  "m_w_2",
            "sma_weather_inj": "m_w_weather_inj",
            "sma_features_inj": "m_w_inj_",
            "tsf": "m_",
            "ifeel": "m_",
            "fourier": "m_"
        },
        "merge_on": ["ean_id", "moy"],  # Merge on ean_id and moy
        "output_suffix": "_m",
        "time_col": "moy"
    },
    "weekly": {
        "prefixes": {
            "sma_weather": "w_weather_2",
            "sma_features": "w_2",
            "sma_weather_inj": "w_weather_inj",
            "sma_features_inj": "w_inj_",
            "ifeel": "w_",
            "fourier": "w_"
        },
        "merge_on": ["ean_id", "week"],
        "output_suffix": "_w",
        "time_col": "week"
    }
}


DATA_CONFIG = {
    "private": {
        "private": True,
        "paths": {
            "data_input": data_root / "Private/Input",
            "data_intermediate": data_root / "Private/Intermediate", 
            "data_smafeatures":  data_root / "Private/FE_SMA",
            "data_tsfeatures":  data_root / "Private/FE_tsfeatures", 
            "data_ifeelfeatures":  data_root / "Private/FE_IFEEL", 
            "data_fourierfeatures":  data_root / "Private/FE_Fourier",
           },
        "schema": {
            "id": "ean_id",
            "datetime": "datetime",
            "consumption": {
                "column": "afname_kw",
                "unit": "kW",
                "interval_minutes": 15,
            },
            "injection": {
                "column": "injectie_kw",   # optional
            },
            "labels": None,
            "technologies": ["HP", "EV", "PV"],
            "subsets": "2024" 
        },
    },
    "public": {
        "private": False,
        "paths": {
            "data_input": data_root / "Public2024/Input",
            "data_intermediate": data_root / "Public2024/Intermediate", 
            "data_smafeatures":  data_root / "Public2024/FE_SMA",
            "data_tsfeatures":  data_root / "Public2024/FE_tsfeatures", 
            "data_ifeelfeatures":  data_root / "Public2024/FE_IFEEL", 
            "data_fourierfeatures":  data_root / "Public2024/FE_Fourier",
           },
        "schema": {
            "id": "EAN_ID",
            "datetime": "Datum_Startuur",
            "consumption": {
                "column": "Volume_Afname_KWh",
                "unit": "kWh",
                "interval_minutes": 15,
            },
            "injection": {
                "column": "Volume_Injectie_KWh",   # optional
            },
            "labels": {"HP": "Warmtepomp_Indicator", "EV": "Elektrisch_Voertuig_Indicator", "PV": "PV_Installatie_Indicator"},  # None OR dict: {"PV": "...", "HP": "..."}            
            "technologies": ["HP", "EV", "PV"],
            "subsets": "2024" 
        },
    },
    "public2022": {
        "paths": {
            "data_input": data_root / "Public2022/Input",
            "data_intermediate": data_root / "Public2022/Intermediate",
            "data_tsfeatures": data_root / "Public2022/FE_tsfeatures",
            "data_smafeatures": data_root / "Public2022/FE_SMA",
            "data_ifeelfeatures": data_root / "Public2022/FE_IFEEL",
            "data_fourierfeatures": data_root /"Public2022/FE_Fourier",
        },
        "schema": {
            "id": "EAN_ID",
            "datetime": "Datum_Startuur",
            "consumption": {
                "column": "Volume_Afname_kWh",
                "unit": "kWh",
                "interval_minutes": 15,
            },
            "injection": {
                "column": "Volume_Injectie_kWh",   # optional
            },
            "labels": {"HP": "Warmtepomp_Indicator", "EV": "Elektrisch_Voertuig_Indicator", "PV": "PV-Installatie_Indicator"},  # None OR dict: {"PV": "...", "HP": "..."}            
            "technologies": ["HP", "EV", "PV"],
            "subsets": "2022" 
        },
    },
    "USER": {
        "paths": {
            "data_input": "", # String: Input data directory
            "data_intermediate": "", # String: Output data directory
            "data_tsfeatures": "", # String: Output data directory
            "data_smafeatures": "", # String: Output data directory
            "data_ifeelfeatures": " ", # String: Output data directory
            "data_fourierfeatures": " ", # String: Output data directory
        },
        "schema": {
            "id": "", # String: Column name (case sensitive)
            "datetime": "", # String: Column name (case sensitive)
            "consumption": {
                "column": "", # String: Column name (case sensitive)
                "unit": "",  # String: kWh or kW (case sensitive)
                "interval_minutes": None, # Numeric value 
            },
            "injection": {
                "column": "",   # Column name (case sensitive) - optional
            },
            "labels": None,  # None OR dict: {"PV": "...", "HP": "..."}    
            "technologies": None, # None OR list ["PV", "HP", "EV"]
            "subsets": "" # String: file prefix for the subsetted files
        },
    },
}

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# PRINT CONFIGURATION
print(f"\n{'='*60}")
print(f"CONFIGURATION: {AGGREGATION_LEVEL.upper()} / {DATA_TYPE.upper()}")
print(f"{'='*60}")

#%% EXECUTE PREPROCESSING 

# ---  1. Weather data: loading and preprocessing ---
weather_files = list((data_root / "Weather").glob("*"))

full_df_weather = []
for f in weather_files:
    if f.suffix == ".xlsx":
        full_df_weather.append(pd.read_excel(f))
    elif f.suffix == ".csv":
        full_df_weather.append(pd.read_csv(f))

weather_raw = pd.concat(full_df_weather, ignore_index=True)
weather = preprocess_weather(weather_raw)

# ---  2. Prices data: loading and preprocessing ---
prices = pd.read_csv(data_root/"Prices/ENTSOE_A44_A01_all.csv")
prices = preprocess_prices(prices)

# ---  3. Metering data: loading and preprocessing --- 
smart_meter_files = list(data_config["paths"]["data_input"].glob("*.csv"))  # or *.xlsx if needed

schema = data_config["schema"]

for file_path in smart_meter_files:
    print(f"Processing {file_path.name}...")
    
    # Load and preprocess Fluvius data
    if file_path.suffix == ".csv":
        fluvius_raw = pd.read_csv(file_path, sep=",")
    else:
        print(f"Skipping unsupported file: {file_path.name}")
        continue

    print(schema["id"])
    
    fluvius = preprocess_electricity_data(fluvius_raw,
        ean_col = schema["id"],
        datetime_col=schema["datetime"],
        afname_col=schema["consumption"]["column"],
        afname_unit=schema["consumption"]["unit"],
        interval_minutes=schema["consumption"]["interval_minutes"],
        injectie_col=schema.get("injection", {}).get("column"),
        labelled_data=schema["labels"] is not None,
        label_cols=schema["labels"],
        )
    
    # Merge with weather
    merged = fluvius.merge(
        weather,
        on=["year", "moy", "day", "hour"],
        how="left"
    )
    
    # Merge with prices
    merged = merged.merge(
        prices,
        on=["year", "moy", "day", "hour"],
        how="left"
    )
    
    out_path = data_config["paths"]["data_intermediate"] / f"{file_path.stem}.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved merged file to {out_path}")

#%% SPLIT DATASET INTO SUBSETS

"""
This subsection is OPTIONAL

 - It splits a single preprocessed file with metering data into separate subfiles, one for each target variable class.
 - It requires columns / column names for technologies to be present, as these determine the splitting criteria. 

"""
def split_by_technologies(df, techs, out_dir, prefix):
    from itertools import product

    if not techs:
        return

    for combo in product([0,1], repeat=len(techs)):
        mask = True
        label = []

        for tech, val in zip(techs, combo):
            mask &= (df[tech] == val)
            if val:
                label.append(tech)

        name = "_".join(label) if label else "NONE"
        subset = df[mask]

        if not subset.empty:
            subset.to_csv(out_dir / f"{prefix}_{name}.csv", index=False)
            print(f"{prefix}_{name}.csv saved")
                     
"""
PUBLIC2024: Data/Public2024/Intermediate - already by technology combination
PRIVATE2024: Data/Private2024/Intermediate - already by technology combination
PUBLIC2022: Data/Private/Intermediate - needs to be splitted by technology
"""

AGGREGATION_LEVEL = "yearly"  
DATA_TYPE = "public2022" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

file_path = list(data_config["paths"]["data_input"].glob("*.csv"))[0]
in_path = data_config["paths"]["data_intermediate"] / f"{file_path.stem}.csv"
df_overall = pd.read_csv(in_path)             
schema = data_config["schema"]

split_by_technologies(
    df_overall,
    schema["technologies"],
    Path(data_config["paths"]["data_intermediate"]),
    prefix=data_config["schema"]["subsets"],
)


#%% FEATURE ENGINEERING - TSFEATURES 
from src.preprocessing_features_tsfeatures import compute_tsfeatures

## 1. List files
subset_files = discover_subset_files(data_config)
print(f'Found {len(subset_files)} subset files')
for f in subset_files:
    print(f"  - {f.name}")


# 2. Yearly tsfeatures
for file in subset_files:
    
    df = pd.read_csv(f'{file}')
    df_yearly = compute_tsfeatures(df = df, freq = 24, agg = "yearly")
    yearly_out = os.path.join(data_config["paths"]["data_tsfeatures"], f"y_{file.name}")
    df_yearly.to_csv(yearly_out, index=False)
    print(f"Saved yearly features → {os.path.basename(yearly_out)}")
    
    
# 3. Monthly features
for file in subset_files:
    
    df = pd.read_csv(f'{file}')
    df_monthly = compute_tsfeatures(df = df, freq = 24, agg = "monthly")
    monthly_out = os.path.join(data_config["paths"]["data_tsfeatures"], f"m_{file.name}")
    df_monthly.to_csv(monthly_out, index=False)
    print(f"Saved monthly features → {os.path.basename(monthly_out)}")
    

#%% FEATURE ENGINEERING - SMA
from src.preprocessing_features_SMA import compute_smafeatures_weekly
from src.preprocessing_features_SMA import aggregate_smafeatures 

## 1. List files
subset_files = discover_subset_files(data_config)
print(f'Found {len(subset_files)} subset files')
for f in subset_files:
    print(f"  - {f.name}")
    
feature = "injectie_kwh" 
feature_abbrev = 'inj'
   
# 2. Weekly SMA
for file in subset_files:
    df = pd.read_csv(f'{file}')
    (df_weekly, df_weekly_weather) = compute_smafeatures_weekly(df = df, feature = feature)
    weekly_out = os.path.join(data_config["paths"]["data_smafeatures"], f"w_{feature_abbrev}_{file.name}")
    weekly_weather_out = os.path.join(data_config["paths"]["data_smafeatures"], f"w_weather_{feature_abbrev}_{file.name}")
    df_weekly.to_csv(weekly_out, index=True)
    df_weekly_weather.to_csv(weekly_weather_out, index=True)
    print("Saved weekly features")
    

# 3. Aggregation
dir_path = Path(data_config["paths"]["data_smafeatures"])
subset_files = sorted(f for f in dir_path.glob(f"w_*{feature_abbrev}_*") if f.is_file())
print(f'Found {len(subset_files)} files starting with "w_":')
for f in subset_files:
    print(f"  - {f.name}")

# 3a. Monthly SMA
for file in subset_files:
    
    df = pd.read_csv(f'{file}')
    df_monthly = aggregate_smafeatures(df = df, agg = "monthly")
    monthly_out = os.path.join(data_config["paths"]["data_smafeatures"], f"m_{file.name}")
    df_monthly.to_csv(monthly_out, index=False)
    print(f"Saved monthly features → {os.path.basename(monthly_out)}")


# 3b. Yearly SMA
for file in subset_files:
    
    df = pd.read_csv(f'{file}')
    df_yearly = aggregate_smafeatures(df = df, agg = "yearly")
    yearly_out = os.path.join(data_config["paths"]["data_smafeatures"], f"y_{file.name}")
    df_yearly.to_csv(yearly_out, index=False)
    print(f"Saved yearly features → {os.path.basename(yearly_out)}")

#%% FEATURE ENGINEERING - IFEEL 
from src.preprocessing_features_IFEEL import compute_ifeel_features

# 1. List files
subset_files = discover_subset_files(data_config)
print(f'Found {len(subset_files)} subset files')
for f in subset_files:
    print(f"  - {f.name}")

# 2. Daily IFEEL including conversion to yearly / monthly / weekly
for file in subset_files:
    df = pd.read_csv(f'{file}')
    
    (df_yearly, df_monthly, df_weekly) = compute_ifeel_features(df = df)
    weekly_out = os.path.join(data_config["paths"]["data_ifeelfeatures"], f"w_{file.name}")
    monthly_out = os.path.join(data_config["paths"]["data_ifeelfeatures"], f"m_{file.name}")
    yearly_out = os.path.join(data_config["paths"]["data_ifeelfeatures"], f"y_{file.name}")

    df_weekly.to_csv(weekly_out, index=False)
    df_monthly.to_csv(monthly_out, index=False)
    df_yearly.to_csv(yearly_out, index=False)

    print(f"Saved features {file.name}")

#%% FEATURE ENGINEERING - FOURIER 
from src.preprocessing_features_Fourier import build_weekly_features_dataframe
from src.preprocessing_features_Fourier import aggregate_fourierfeatures

# 1. List files
subset_files = discover_subset_files(data_config)
print(f'Found {len(subset_files)} subset files')
for f in subset_files:
    print(f"  - {f.name}")

# 2. Weekly Fourier
for file in subset_files:
    df = pd.read_csv(f'{file}')
    (df_weekly) = build_weekly_features_dataframe(df = df)
    weekly_out = os.path.join(data_config["paths"]["data_fourierfeatures"], f"w_{file.name}")
    df_weekly.to_csv(weekly_out, index=False)
    print("Saved weekly features")
    

# 3. Aggregation
dir_path = Path(data_config["paths"]["data_fourierfeatures"])
subset_files = sorted(f for f in dir_path.glob("w_*") if f.is_file())
print(f'Found {len(subset_files)} files starting with "w_":')
for f in subset_files:
    print(f"  - {f.name}")

# 3a. Monthly Fourier
for file in subset_files:
    
    df = pd.read_csv(f'{file}')
    df_monthly = aggregate_fourierfeatures(df = df, agg = "monthly")
    monthly_out = os.path.join(data_config["paths"]["data_fourierfeatures"], f"m_{file.name}")
    df_monthly.to_csv(monthly_out, index=False)
    print("Saved monthly features")


# 3b. Yearly Fourier
for file in subset_files:
    
    df = pd.read_csv(f'{file}')
    df_yearly = aggregate_fourierfeatures(df = df, agg = "yearly")
    yearly_out = os.path.join(data_config["paths"]["data_fourierfeatures"], f"y_{file.name}")
    df_yearly.to_csv(yearly_out, index=False)
    print("Saved yearly features")