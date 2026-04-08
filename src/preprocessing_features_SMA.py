import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import r, packages, FloatVector, rinterface
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
smart = packages.importr("SmartMeterAnalytics")

################################################
# Metering features
def calc_features15_py(B: np.ndarray,
                        rowname: str = None,
                        featsCoarserGranularity: bool = False,
                        replace_NA_with_defaults: bool = True) -> pd.DataFrame:
    if len(B) != 672:
        raise ValueError(f"B must have length 672 (got {len(B)})")

    r_B = FloatVector(B.tolist())
    r_rowname = rinterface.NULL if rowname is None else rowname

    calc_features = r['calc_features15_consumption']
    result = calc_features(
        B=r_B,
        rowname=r_rowname,
        featsCoarserGranularity=featsCoarserGranularity,
        replace_NA_with_defaults=replace_NA_with_defaults
    )

    # Convert R object to pandas DataFrame cleanly
    with localconverter(pandas2ri.converter):
        py_obj = robjects.conversion.rpy2py(result)

    # Flatten in case it's nested
    if isinstance(py_obj, pd.DataFrame):
        df = py_obj
    elif isinstance(py_obj, np.ndarray):
        df = pd.DataFrame(py_obj.reshape(1, -1), columns=result.names)
    else:
        df = pd.DataFrame([list(py_obj)], columns=result.names)

    if rowname:
        df.insert(0, 'rowname', rowname)

    return df

################################################
# Weather features
def calc_features_weather_py(B: np.ndarray,
                             WEATHER: np.ndarray,
                             rowname: str = None) -> pd.DataFrame:
    
    # --- Input checks ---
    if len(B) not in (336, 672):
        raise ValueError(f"B must have length 336 or 672 (got {len(B)}).")

    if len(WEATHER) != 336:
        raise ValueError(f"WEATHER must have length 336 (got {len(WEATHER)}).")

    # --- Convert to R vectors ---
    r_B = FloatVector(B.tolist())
    r_WEATHER = FloatVector(WEATHER.tolist())
    r_rowname = rinterface.NULL if rowname is None else rowname

    # --- Call the R function ---
    calc_weather = r['calc_features_weather']
    result = calc_weather(
        SMD=r_B,          
        WEATHER=r_WEATHER,
        rowname=r_rowname
    )

    # --- Convert back to pandas ---
    with localconverter(pandas2ri.converter):
        py_obj = robjects.conversion.rpy2py(result)

    # --- Normalize output format ---
    if isinstance(py_obj, pd.DataFrame):
        df = py_obj
    elif isinstance(py_obj, np.ndarray):
        df = pd.DataFrame(py_obj.reshape(1, -1), columns=result.names)
    else:
        df = pd.DataFrame([list(py_obj)], columns=result.names)

    # Insert rowname if provided
    if rowname is not None:
        df.insert(0, "rowname", rowname)

    return df

################################################
def compute_smafeatures_weekly(df):

    ## PART 1: ELECTRICITY CONSUMPTION FEATURES
    # --- Convert datetime & aggregate hourly ---
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # --- Rename columns ---
    df = df.rename(columns={'EAN_ID': 'unique_id','datetime': 'ds', 'afname_kwh': 'y'})
    
    
    results = []
    
    for uid in df['unique_id'].unique():
        for w in df['week'].unique():
            
            subset = df[(df['unique_id'] == uid) & (df['week'] == w) & (df['year'] == 2022)]
            
            y_vector = subset['y'].values
            
            # Check length
            if len(y_vector) != 672:
                # Create NaN row with proper feature names
                # We can use calc_features15_py on a dummy vector to get feature names
                try:
                    feature_names = calc_features15_py(np.random.rand(672)).columns.tolist()
                except Exception:
                    feature_names = [f"feat_{i}" for i in range(62)]  # fallback
                    
                nan_row = pd.Series([np.nan] * len(feature_names), index=feature_names)
                nan_row['unique_id'] = uid
                nan_row['week'] = w
                results.append(nan_row)
                print(f"Failed {uid} week {w}")
                continue
            
            # Compute features
            df_features = calc_features15_py(y_vector, rowname=None)
            df_features['unique_id'] = uid
            df_features['week'] = w
            results.append(df_features.iloc[0])  # add as Series
            print(f"Success {uid} week {w}")
    
    # Combine all rows into a single DataFrame
    df_all_features = pd.DataFrame(results)
    
    # Set index
    df_all_features.set_index(['unique_id', 'week'], inplace=True)
        
    ## PART 2: WEATHER CORRELATION FEATURES
    results = []
    for uid in df['unique_id'].unique():
        for w in df['week'].unique():
            
            subset = df[(df['unique_id'] == uid) & (df['week'] == w) & (df['year'] == 2022)]
            
            y_vector = subset['y'].values
            
            subset_W = df[(df['unique_id'] == uid) & (df['week'] == w) & (df['year'] == 2022) & (df['min'].isin([0, 30]))]

            WEATHER =  subset_W['T'].values           

            # Check length
            if len(y_vector) != 672:
                # Create NaN row with proper feature names
                # We can use calc_features_weather_py on a dummy vector to get feature names
                try:
                    feature_names = calc_features_weather_py(np.random.rand(672)).columns.tolist()
                except Exception:
                    feature_names = [f"feat_{i}" for i in range(62)]  # fallback
                    
                nan_row = pd.Series([np.nan] * len(feature_names), index=feature_names)
                nan_row['unique_id'] = uid
                nan_row['week'] = w
                results.append(nan_row)
                print(f"Failed {uid} week {w}")
                continue
            
            # Compute features
            df_features = calc_features_weather_py(y_vector, WEATHER, rowname=None)
            df_features['unique_id'] = uid
            df_features['week'] = w
            results.append(df_features.iloc[0])  # add as Series
            print(f"Success {uid} week {w}")
    
    # Combine all rows into a single DataFrame
    df_weather_features = pd.DataFrame(results)
    
    # Set index
    df_weather_features.set_index(['unique_id', 'week'], inplace=True)

    return (df_all_features, df_weather_features)


################################################
def aggregate_smafeatures(df, agg):
 
    if agg == "yearly":
        
        feature_cols = [c for c in df.columns if c not in ['unique_id', 'week']]
    
        collapsed_year = (
            df.groupby(['unique_id'])[feature_cols]
            .agg(['min', 'max', 'median', 'mean', 'std'])
            .reset_index()
        )
    
        collapsed_year.columns = [
            '_'.join(filter(None, col)).strip() for col in collapsed_year.columns.to_flat_index()
        ]
        
        return collapsed_year
    
    if agg == "monthly":
        
        moy_to_weeks = {
            1:  [1,2,3,4,5],
            2:  [6,7,8,9],
            3:  [10,11,12,13],
            4:  [14,15,16,17,18],
            5:  [19,20,21,22],
            6:  [23,24,25,26],
            7:  [27,28,29,30,31],
            8:  [32,33,34,35],
            9:  [36,37,38,39,40],
            10: [41,42,43,44],
            11: [45,46,47,48],
            12: [49,50,51,52]
        }

        week_to_moy = {w: m for m, weeks in moy_to_weeks.items() for w in weeks}
    
        df["moy"] = df["week"].map(week_to_moy)

        feature_cols = [c for c in df.columns if c not in ['unique_id', 'week', 'moy']]

        collapsed_month = (
            df.groupby(['unique_id', 'moy'])[feature_cols]
            .agg(['min', 'max', 'median', 'mean', 'std'])
            .reset_index()
        )

        collapsed_month.columns = [
            '_'.join(filter(None, col)).strip() for col in collapsed_month.columns.to_flat_index()
        ]
        
        return collapsed_month

# #%% Preliminaries


# """
# This script computes weekly, monthly, and yearly features for several Fluvius
# smart meter datasets using the R package `SmartMeterAnalytics` (via rpy2).

# 1. INPUTS
    
#     - config.yml
#         Contains paths to directories. 
    
#     - Intermediate/{file}.csv
#         Columns required:
#             - EAN_ID      : household identifier
#             - datetime    : timestamp at 15-minute frequency
#             - afname_kwh  : consumption in kWh (15 min)
#             - week        : identifier for the week
#             - year        : identifier for the year
#             - T           : temperature
            

# 2. PROCESSING STEPS

#     A) Electricity-only SMA features (calc_features15_consumption)
#        → For each household × week (year=2024), compute 672-point weekly profiles.
#        → Output: {file}_SMA_w.csv
    
#     B) Weather-augmented SMA features (calc_features_weather)
#        → Same loop, but also include matching 336-point weather temperature series.
#        → Output: {file}_SMA_WEATHER_w.csv
    
#     C) Yearly aggregation
#        → Collapse weekly features to yearly statistics (min/max/median/mean/std).
#        → Output: {file}_colly.csv
    
#     D) Monthly aggregation
#        → Convert week → month (moy) mapping.
#        → Collapse weekly features within each month and compute monthly statistics (min/max/median/mean/std).
#        → Output: {file}_collm.csv


# 3. OUTPUTS (directory = Intermediate/FE_SMA)

#     - Weekly SMA features:
#           {file}_SMA_w.csv
#           {file}_SMA_WEATHER_w.csv
    
#     - Monthly features:
#           {file}_SMA_w_collm.csv
#           {file}_SMA_WEATHER_w_collm.csv
              
#     - Yearly features:
#           {file}_SMA_w_colly.csv
#           {file}_SMA_WEATHER_w_colly.csv

# """


# import os
# import yaml
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import rpy2.robjects as robjects
# from rpy2.robjects import r, packages, FloatVector, rinterface
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects import pandas2ri
# from datetime import datetime, time

# with open("config.yml") as f:
#     config = yaml.safe_load(f)
    
# data_root = Path(config["data_root"])
# data_root_intermediate = data_root / "Intermediate"
# data_root_sma = data_root / "Intermediate/FE_SMA"
# data_root_intermediate_private = data_root / "Private/Intermediate"
# data_root_sma_private = data_root / "Private/FE_SMA"
# data_root_intermediate_2022 = data_root / "Public2022/Intermediate"
# data_root_sma_2022 = data_root / "Public2022/FE_SMA"


# #%% 1. Define helper function  and R preliminaries
# # r('R.version.string')
# # r('install.packages("lattice", repos="https://cloud.r-project.org")')

# # Load R package globally
# smart = packages.importr("SmartMeterAnalytics")

# ################################################
# # Metering features
# def calc_features15_py(B: np.ndarray,
#                         rowname: str = None,
#                         featsCoarserGranularity: bool = False,
#                         replace_NA_with_defaults: bool = True) -> pd.DataFrame:
#     if len(B) != 672:
#         raise ValueError(f"B must have length 672 (got {len(B)})")

#     r_B = FloatVector(B.tolist())
#     r_rowname = rinterface.NULL if rowname is None else rowname

#     calc_features = r['calc_features15_consumption']
#     result = calc_features(
#         B=r_B,
#         rowname=r_rowname,
#         featsCoarserGranularity=featsCoarserGranularity,
#         replace_NA_with_defaults=replace_NA_with_defaults
#     )

#     # Convert R object to pandas DataFrame cleanly
#     with localconverter(pandas2ri.converter):
#         py_obj = robjects.conversion.rpy2py(result)

#     # Flatten in case it's nested
#     if isinstance(py_obj, pd.DataFrame):
#         df = py_obj
#     elif isinstance(py_obj, np.ndarray):
#         df = pd.DataFrame(py_obj.reshape(1, -1), columns=result.names)
#     else:
#         df = pd.DataFrame([list(py_obj)], columns=result.names)

#     if rowname:
#         df.insert(0, 'rowname', rowname)

#     return df

# ################################################
# # Weather features
# def calc_features_weather_py(B: np.ndarray,
#                              WEATHER: np.ndarray,
#                              rowname: str = None) -> pd.DataFrame:
    
#     # --- Input checks ---
#     if len(B) not in (336, 672):
#         raise ValueError(f"B must have length 336 or 672 (got {len(B)}).")

#     if len(WEATHER) != 336:
#         raise ValueError(f"WEATHER must have length 336 (got {len(WEATHER)}).")

#     # --- Convert to R vectors ---
#     r_B = FloatVector(B.tolist())
#     r_WEATHER = FloatVector(WEATHER.tolist())
#     r_rowname = rinterface.NULL if rowname is None else rowname

#     # --- Call the R function ---
#     calc_weather = r['calc_features_weather']
#     result = calc_weather(
#         SMD=r_B,          
#         WEATHER=r_WEATHER,
#         rowname=r_rowname
#     )

#     # --- Convert back to pandas ---
#     with localconverter(pandas2ri.converter):
#         py_obj = robjects.conversion.rpy2py(result)

#     # --- Normalize output format ---
#     if isinstance(py_obj, pd.DataFrame):
#         df = py_obj
#     elif isinstance(py_obj, np.ndarray):
#         df = pd.DataFrame(py_obj.reshape(1, -1), columns=result.names)
#     else:
#         df = pd.DataFrame([list(py_obj)], columns=result.names)

#     # Insert rowname if provided
#     if rowname is not None:
#         df.insert(0, "rowname", rowname)

#     return df


# #%% 2. Compute SMA features with a loop over the metering data. 

# # PUBLIC
# # csv_files = [
# #     "P6269_Open_Data_enkel_ZP_merged",
# #     "P6269_Open_Data_EV_geen_ZP_merged",
# #     "P6269_Open_Data_EV_met_ZP_merged",
# #     "P6269_Open_Data_geen_ZP_merged",
# #     "P6269_Open_Data_WP_EV_geen_ZP_merged",
# #     "P6269_Open_Data_WP_EV_met_ZP_merged",
# #     "P6269_Open_Data_WP_geen_ZP_merged",
# #     "P6269_Open_Data_WP_met_ZP_merged"
# # ]


# # PRIVATE
# # csv_files = [
# #     "2024_EV_merged",
# #     "2024_EVHP_merged",
# #     "2024_HP_merged",
# #     "2024_NONE_merged",
# #     "2024_PV_merged",
# #     "2024_PVEV_merged",
# #     "2024_PVEVHP_merged",
# #     "2024_PVHP_merged"
# # ]


# # PUBLIC 2022
# csv_files = [
#     "2022_EV",
#     "2022_NONE",
#     "2022_PV",
#     "2022_PVEV",
#     "2022_PVHP"
# ]


# ################################################
# # Electricity consumption profile features
# for file in csv_files:
    
#     df = pd.read_csv(f"{data_root_intermediate_2022}\\{file}.csv")

#     # df = pd.read_csv(f"{data_root_intermediate}\\{file}.csv")
    
#     # --- Convert datetime & aggregate hourly ---
#     df['datetime'] = pd.to_datetime(df['datetime'])
    
     
#     # --- Rename columns ---
#     df = df.rename(columns={
#         'EAN_ID': 'unique_id',
#         'datetime': 'ds',
#         'afname_kwh': 'y'
#     })
    
    
#     results = []
    
#     for uid in df['unique_id'].unique():
#         for w in df['week'].unique():
            
#             subset = df[(df['unique_id'] == uid) & (df['week'] == w) & (df['year'] == 2022)]
            
#             y_vector = subset['y'].values
            
#             # Check length
#             if len(y_vector) != 672:
#                 # Create NaN row with proper feature names
#                 # We can use calc_features15_py on a dummy vector to get feature names
#                 try:
#                     feature_names = calc_features15_py(np.random.rand(672)).columns.tolist()
#                 except Exception:
#                     feature_names = [f"feat_{i}" for i in range(62)]  # fallback
                    
#                 nan_row = pd.Series([np.nan] * len(feature_names), index=feature_names)
#                 nan_row['unique_id'] = uid
#                 nan_row['week'] = w
#                 results.append(nan_row)
#                 print(f"Failed {uid} week {w}")
#                 continue
            
#             # Compute features
#             df_features = calc_features15_py(y_vector, rowname=None)
#             df_features['unique_id'] = uid
#             df_features['week'] = w
#             results.append(df_features.iloc[0])  # add as Series
#             print(f"Success {uid} week {w}")
    
#     # Combine all rows into a single DataFrame
#     df_all_features = pd.DataFrame(results)
    
#     # Set index
#     df_all_features.set_index(['unique_id', 'week'], inplace=True)
    
#     print(df_all_features.head())
    
#     path = os.path.join(data_root_sma_2022, f"{file}_SMA_w.csv")
#     df_all_features.to_csv(path, index=True)
#     print(f"Saved weekly: {file}_SMA_w")

# ################################################
# # Weather related features
# for file in csv_files:

#     # df = pd.read_csv(f"{data_root_intermediate}\\{file}.csv")
#     df = pd.read_csv(f"{data_root_intermediate_2022}\\{file}.csv")
    
#     # --- Convert datetime & aggregate hourly ---
#     df['datetime'] = pd.to_datetime(df['datetime'])
    
     
#     # --- Rename columns  ---
#     df = df.rename(columns={
#         'EAN_ID': 'unique_id',
#         'datetime': 'ds',
#         'afname_kwh': 'y'
#     })
    
    
#     results = []
    
#     for uid in df['unique_id'].unique():
#         for w in df['week'].unique():
            
#             subset = df[(df['unique_id'] == uid) & (df['week'] == w) & (df['year'] == 2022)]
            
#             y_vector = subset['y'].values
            
#             subset_W = df[(df['unique_id'] == uid) & (df['week'] == w) & (df['year'] == 2022) & (df['min'].isin([0, 30]))]

#             WEATHER =  subset_W['T'].values           

#             # Check length
#             if len(y_vector) != 672:
#                 # Create NaN row with proper feature names
#                 # We can use calc_features_weather_py on a dummy vector to get feature names
#                 try:
#                     feature_names = calc_features_weather_py(np.random.rand(672)).columns.tolist()
#                 except Exception:
#                     feature_names = [f"feat_{i}" for i in range(62)]  # fallback
                    
#                 nan_row = pd.Series([np.nan] * len(feature_names), index=feature_names)
#                 nan_row['unique_id'] = uid
#                 nan_row['week'] = w
#                 results.append(nan_row)
#                 print(f"Failed {uid} week {w}")
#                 continue
            
#             # Compute features
#             df_features = calc_features_weather_py(y_vector, WEATHER, rowname=None)
#             df_features['unique_id'] = uid
#             df_features['week'] = w
#             results.append(df_features.iloc[0])  # add as Series
#             print(f"Success {uid} week {w}")
    
#     # Combine all rows into a single DataFrame
#     df_all_features = pd.DataFrame(results)
    
#     # Set index
#     df_all_features.set_index(['unique_id', 'week'], inplace=True)
    
#     print(df_all_features.head())
    
#     #path = os.path.join(data_root_sma_private, f"{file}_SMA_WEATHER_w.csv")
#     path = os.path.join(data_root_sma_2022, f"{file}_SMA_WEATHER_w.csv")
#     df_all_features.to_csv(path, index=True)
#     print(f"Saved weekly: {file}_SMA_WEATHER_w")

# #%% Conversion to yearly dataset

# csv_files = [
#     "P6269_Open_Data_enkel_ZP_merged_SMA_w",
#     "P6269_Open_Data_EV_geen_ZP_merged_SMA_w",
#     "P6269_Open_Data_EV_met_ZP_merged_SMA_w",
#     "P6269_Open_Data_geen_ZP_merged_SMA_w",
#     "P6269_Open_Data_WP_EV_geen_ZP_merged_SMA_w",
#     "P6269_Open_Data_WP_EV_met_ZP_merged_SMA_w",
#     "P6269_Open_Data_WP_geen_ZP_merged_SMA_w",
#     "P6269_Open_Data_WP_met_ZP_merged_SMA_w",
#     "P6269_Open_Data_enkel_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_EV_geen_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_EV_met_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_geen_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_WP_EV_geen_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_WP_EV_met_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_WP_geen_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_WP_met_ZP_merged_SMA_WEATHER_w"
# ]

# csv_files = [
#     "2024_EV_merged_SMA_w",
#     "2024_EVHP_merged_SMA_w",
#     "2024_HP_merged_SMA_w",
#     "2024_NONE_merged_SMA_w",
#     "2024_PV_merged_SMA_w",
#     "2024_PVEV_merged_SMA_w",
#     "2024_PVEVHP_merged_SMA_w",
#     "2024_PVHP_merged_SMA_w",
#     "2024_EV_merged_SMA_WEATHER_w",
#     "2024_EVHP_merged_SMA_WEATHER_w",
#     "2024_HP_merged_SMA_WEATHER_w",
#     "2024_NONE_merged_SMA_WEATHER_w",
#     "2024_PV_merged_SMA_WEATHER_w",
#     "2024_PVEV_merged_SMA_WEATHER_w",
#     "2024_PVEVHP_merged_SMA_WEATHER_w",
#     "2024_PVHP_merged_SMA_WEATHER_w"
# ]


# csv_files = [
#     "2022_EV_SMA_w",
#     "2022_NONE_SMA_w",
#     "2022_PV_SMA_w",
#     "2022_PVEV_SMA_w",
#     "2022_PVHP_SMA_w",
#     "2022_EV_SMA_WEATHER_w",
#     "2022_NONE_SMA_WEATHER_w",
#     "2022_PV_SMA_WEATHER_w",
#     "2022_PVEV_SMA_WEATHER_w",
#     "2022_PVHP_SMA_WEATHER_w",
# ]


# for file in csv_files:
        
#     df = pd.read_csv(f"{data_root_sma_2022}\\{file}.csv")
#     print(f"Loaded {file}")
    
#     feature_cols = [c for c in df.columns if c not in ['unique_id', 'week']]

#     collapsed_year = (
#         df.groupby(['unique_id'])[feature_cols]
#         .agg(['min', 'max', 'median', 'mean', 'std'])
#         .reset_index()
#     )

#     collapsed_year.columns = [
#         '_'.join(filter(None, col)).strip() for col in collapsed_year.columns.to_flat_index()
#     ]
    
#     path = os.path.join(data_root_sma_2022, f"{file}_colly.csv")
#     collapsed_year.to_csv(path, index=False)
#     print(f"Saved yearly: {file}")

    
# #%% Conversion to monthly dataset
# moy_to_weeks = {
#     1:  [1,2,3,4,5],
#     2:  [6,7,8,9],
#     3:  [10,11,12,13],
#     4:  [14,15,16,17,18],
#     5:  [19,20,21,22],
#     6:  [23,24,25,26],
#     7:  [27,28,29,30,31],
#     8:  [32,33,34,35],
#     9:  [36,37,38,39,40],
#     10: [41,42,43,44],
#     11: [45,46,47,48],
#     12: [49,50,51,52]
# }

# week_to_moy = {w: m for m, weeks in moy_to_weeks.items() for w in weeks}

# csv_files = [
#     "P6269_Open_Data_enkel_ZP_merged_SMA_w",
#     "P6269_Open_Data_EV_geen_ZP_merged_SMA_w",
#     "P6269_Open_Data_EV_met_ZP_merged_SMA_w",
#     "P6269_Open_Data_geen_ZP_merged_SMA_w",
#     "P6269_Open_Data_WP_EV_geen_ZP_merged_SMA_w",
#     "P6269_Open_Data_WP_EV_met_ZP_merged_SMA_w",
#     "P6269_Open_Data_WP_geen_ZP_merged_SMA_w",
#     "P6269_Open_Data_WP_met_ZP_merged_SMA_w",
#     "P6269_Open_Data_enkel_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_EV_geen_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_EV_met_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_geen_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_WP_EV_geen_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_WP_EV_met_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_WP_geen_ZP_merged_SMA_WEATHER_w",
#     "P6269_Open_Data_WP_met_ZP_merged_SMA_WEATHER_w"
# ]


# csv_files = [
#     "2024_EV_merged_SMA_w",
#     "2024_EVHP_merged_SMA_w",
#     "2024_HP_merged_SMA_w",
#     "2024_NONE_merged_SMA_w",
#     "2024_PV_merged_SMA_w",
#     "2024_PVEV_merged_SMA_w",
#     "2024_PVEVHP_merged_SMA_w",
#     "2024_PVHP_merged_SMA_w",
#     "2024_EV_merged_SMA_WEATHER_w",
#     "2024_EVHP_merged_SMA_WEATHER_w",
#     "2024_HP_merged_SMA_WEATHER_w",
#     "2024_NONE_merged_SMA_WEATHER_w",
#     "2024_PV_merged_SMA_WEATHER_w",
#     "2024_PVEV_merged_SMA_WEATHER_w",
#     "2024_PVEVHP_merged_SMA_WEATHER_w",
#     "2024_PVHP_merged_SMA_WEATHER_w"
# ]

# csv_files = [
#     "2022_EV_SMA_w",
#     "2022_NONE_SMA_w",
#     "2022_PV_SMA_w",
#     "2022_PVEV_SMA_w",
#     "2022_PVHP_SMA_w",
#     "2022_EV_SMA_WEATHER_w",
#     "2022_NONE_SMA_WEATHER_w",
#     "2022_PV_SMA_WEATHER_w",
#     "2022_PVEV_SMA_WEATHER_w",
#     "2022_PVHP_SMA_WEATHER_w",
# ]




# for file in csv_files:
        
#     df = pd.read_csv(f"{data_root_sma_2022}\\{file}.csv")
#     print(f"Loaded {file}")
    
#     df["moy"] = df["week"].map(week_to_moy)

#     feature_cols = [c for c in df.columns if c not in ['unique_id', 'week', 'moy']]

#     collapsed_month = (
#         df.groupby(['unique_id', 'moy'])[feature_cols]
#         .agg(['min', 'max', 'median', 'mean', 'std'])
#         .reset_index()
#     )

#     collapsed_month.columns = [
#         '_'.join(filter(None, col)).strip() for col in collapsed_month.columns.to_flat_index()
#     ]
    
#     path = os.path.join(data_root_sma_2022, f"{file}_collm.csv")
#     collapsed_month.to_csv(path, index=False)
#     print(f"Saved monthly: {file}")


