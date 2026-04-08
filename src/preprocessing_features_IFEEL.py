import os
import yaml
import string
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, time
from scipy.stats import norm, skew, kurtosis

def ifeel_feature_transformation(df_test, alphabet_size, time_business_start, time_business_end):
    
    # SAX helper functions
    def discretizer(ts, breakpoints):
        return np.where(breakpoints > float(ts))[0][0]

    def stringizer(row):
        return ''.join(string.ascii_letters[int(row['level'])])

    # Add "is_business_hour" to columns as MultiIndex
    is_busi_hour_all = []
    for time_str in df_test.columns:
        t_test = datetime.strptime(time_str, '%H:%M:%S').time()
        is_busi_hour = time_business_start <= t_test.hour <= time_business_end
        is_busi_hour_all.append(is_busi_hour)

    tuples = list(zip(df_test.columns, is_busi_hour_all))
    df_test.columns = pd.MultiIndex.from_tuples(tuples, names=('Time', 'business_hour'))

    # SAX representation
    znorm_list = []
    SAX_number_list = []
    SAX_letter_list = []

    breakpoints = norm.ppf(np.linspace(1. / alphabet_size, 1 - 1. / alphabet_size, alphabet_size - 1))
    breakpoints = np.concatenate((breakpoints, np.array([np.inf])))

    for i in range(df_test.shape[0]):
        row = df_test.iloc[i].fillna(method='ffill')
        y = (row - row.mean()) / row.std()
        y = pd.DataFrame(y.values, columns=["normalized_power"])
        y['level'] = y.apply(discretizer, axis=1, args=[breakpoints])
        y['letter'] = y.apply(stringizer, axis=1)

        znorm_list.append(y['normalized_power'])
        SAX_number_list.append(y['level'])
        SAX_letter_list.append(y['letter'])

    df_znorm_all_house = pd.concat(znorm_list, axis=1).T
    df_SAX_number_all_house = pd.concat(SAX_number_list, axis=1).T
    df_SAX_all_house = pd.concat(SAX_letter_list, axis=1).T

    df_znorm_all_house.index = df_test.index
    df_SAX_number_all_house.index = df_test.index
    df_SAX_all_house.index = df_test.index

    df_znorm_all_house.columns = df_test.columns
    df_SAX_number_all_house.columns = df_test.columns
    df_SAX_all_house.columns = df_test.columns

    # Differences for downstream features
    df_SAX_number_diff_pivot = df_SAX_number_all_house.diff(periods=1, axis=1)
    df_raw_diff = df_test.diff(periods=1, axis=1)

    return df_test, df_raw_diff, df_SAX_number_all_house, df_SAX_all_house, df_SAX_number_diff_pivot


## Manual execution: ifeel_feature_global function


# === GLOBAL FEATURE NAMES ===
feature_name_global = [
    'Mean',
    'Std',
    'Max',
    'Min',
    'Range (i.e., max-min)',
    'Percentage above mean',
    'Sum of net loads during business hours',
    'Sum of net loads during non-business hours',
    'Skewness',
    'Kurtosis',
    'Mode of 5-bin histogram',
    'Longest period above mean',
    'Longest period of successive increase'
]

feature_name_peak = [
    'Peak_all: number',
    'Peak_all: time',
    'Peak_all: shortest interval between two peaks',
    'Peak_all: duration',
    'Peak_longest: occurrence time',
    'Peak_longest: duration',
    'Peak_longest: upward slope',
    'Peak_longest: downward slope'
]


def _get_length_sequences_where(x):
    """Return list of lengths of consecutive True/1 sequences in x."""
    if len(x) == 0:
        return [0]
    res = [len(list(g)) for v, g in itertools.groupby(x) if v == 1]
    return res if len(res) > 0 else [0]


# === CLASS: GLOBAL FEATURES ===
class ifeel_feature_global:
    def __init__(self, ts, ts_diff, sample_interval):
        self.ts = ts
        self.ts_diff = ts_diff
        self.sample_interval = sample_interval

    def global_mean(self): return np.mean(self.ts)
    def global_std(self): return np.std(self.ts)
    def global_max(self): return np.max(self.ts)
    def global_min(self): return np.min(self.ts)
    def global_range(self): return np.max(self.ts) - np.min(self.ts)

    def global_percentage_above_mean(self):
        return np.sum(self.ts > np.mean(self.ts)) / len(self.ts)

    def global_sum_net_loads_busi(self):
        ts_busi = self.ts.loc[self.ts.index.get_level_values('business_hour') == True]
        return ts_busi.sum()

    def global_sum_net_loads_nonbusi(self):
        ts_nonbusi = self.ts.loc[self.ts.index.get_level_values('business_hour') == False]
        return ts_nonbusi.sum()

    def global_skewness(self): return skew(self.ts)
    def global_kurtosis(self): return kurtosis(self.ts)

    def global_mode_histogram_bins5(self):
        x = self.ts.values if isinstance(self.ts, pd.Series) else np.asarray(self.ts)
        if np.all(np.isnan(x)):
            return np.nan  # or 0 if you prefer
        hist, edges = np.histogram(x[~np.isnan(x)], bins=5)  # ignore NaNs
        idx_max = np.argmax(hist)
        mode_range = [edges[idx_max], edges[idx_max + 1]]
        return np.mean(mode_range)

    def global_longest_period_above_mean(self):
        x = np.asarray(self.ts)
        if x.size == 0:
            return 0
        return self.sample_interval * np.max(_get_length_sequences_where(x > np.mean(x)))

    def global_longest_period_of_successive_increase(self):
        x = np.asarray(self.ts_diff)
        if x.size == 0:
            return 0
        return self.sample_interval * np.max(_get_length_sequences_where(x > 0))

    def global_all(self):
        """Return all 13 global features as a labeled Series."""
        data = [
            self.global_mean(),
            self.global_std(),
            self.global_max(),
            self.global_min(),
            self.global_range(),
            self.global_percentage_above_mean(),
            self.global_sum_net_loads_busi(),
            self.global_sum_net_loads_nonbusi(),
            self.global_skewness(),
            self.global_kurtosis(),
            self.global_mode_histogram_bins5(),
            self.global_longest_period_above_mean(),
            self.global_longest_period_of_successive_increase()
        ]
        return pd.Series(data, index=feature_name_global)


# === FUNCTION: PEAK FEATURES ===
def ifeel_feature_peak_period(ts_sax_number, ts_sax_number_diff, alphabet_size, sample_interval):
    """
    Extracts peak-related features from a symbolic (SAX) time series.
    Returns a labeled Series ready to combine into a DataFrame.
    """
    peak = alphabet_size - 1
    ts_to_boolean = ts_sax_number == peak
    peak_index_connected = np.where(ts_to_boolean)[0]

    is_peak_exist = float(peak) in ts_sax_number.values
    if not is_peak_exist:
        # No peak detected
        features = {
            "peak_number": 0,
            "peak_time": np.nan,
            "peak_time_diff_shortest": np.nan,
            "peak_duration": np.nan,
            "peak_longest_time": np.nan,
            "peak_longest_duration": np.nan,
            "peak_longest_slope_upward": np.nan,
            "peak_longest_slope_downward": np.nan
        }
        return pd.Series(features)

    # Group consecutive indices (each group = one peak)
    peak_index_separated = np.split(
        peak_index_connected,
        np.where(np.diff(peak_index_connected) != 1)[0] + 1
    )

    peak_number = len(peak_index_separated)
    peak_time = np.array([np.mean(i) for i in peak_index_separated]) * sample_interval

    # Shortest time difference between peaks
    peak_time_diff_shortest = (
        np.nan if peak_number == 1 else np.min(np.diff(peak_time)) * sample_interval
    )

    # Duration (length × sample interval)
    peak_duration_points = np.array([len(i) for i in peak_index_separated])
    peak_duration = peak_duration_points * sample_interval

    # Longest peak
    longest_len = max(peak_duration_points)
    peak_longest_sequence = next(i for i in peak_index_separated if len(i) == longest_len)
    peak_longest_time = np.mean(peak_longest_sequence) * sample_interval
    peak_longest_duration = longest_len * sample_interval

    # Slopes
    peak_longest_slope_upward = ts_sax_number_diff.iloc[peak_longest_sequence[0]]

    last_pos = len(ts_sax_number_diff) - 1
    end_idx = peak_longest_sequence[-1]
    
    if end_idx >= last_pos:
        # we're at or past the last point in the series -> no downward slope available
        peak_longest_slope_downward = -1
    else:
        peak_longest_slope_downward = ts_sax_number_diff.iloc[end_idx + 1]

    # Build labeled Series
    features = {
        "peak_number": peak_number,
        "peak_time": peak_time,
        "peak_time_diff_shortest": peak_time_diff_shortest,
        "peak_duration": peak_duration,
        "peak_longest_time": peak_longest_time,
        "peak_longest_duration": peak_longest_duration,
        "peak_longest_slope_upward": peak_longest_slope_upward,
        "peak_longest_slope_downward": peak_longest_slope_downward
    }

    return pd.Series(features)




def compute_ifeel_features(df):
    
    
    # --- Convert datetime & aggregate hourly ---
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = (
         df.groupby(['EAN_ID', pd.Grouper(key='datetime', freq='H')], as_index=False)
              .agg({'afname_kwh': 'sum'})
    )
         
    # --- Rename columns for tsfeatures ---
    df = df.rename(columns={
        'EAN_ID': 'unique_id',
        'datetime': 'ds',
        'afname_kwh': 'y'
    })
         
    time_business_start = 9
    time_business_end = 17
    alphabet_size = 7 
    sample_interval_in_hour = 1
    
    
    all_global_features = []
    all_peak_features = []
    
    # --- DAILY FEATURES ---
    for uid in df['unique_id'].unique():
        
        # Create a smaller DataFrame for this household
        sub_df = df[df['unique_id'] == uid].copy()
        
        sub_df['ds'] = pd.to_datetime(sub_df['ds'])
        sub_df['date'] = sub_df['ds'].dt.date
        sub_df['hour'] = sub_df['ds'].dt.hour
        
        all_features = []
        
        df_wide = sub_df.pivot(index='date', columns='hour', values='y')
        df_wide = df_wide.sort_index(axis=1)  # ensure columns in hour order
        df_wide.columns = [f"{int(c):02d}:00:00" for c in df_wide.columns]
        df_wide = df_wide.fillna(0)
        
        
        try:
            [df_raw, df_raw_diff, df_SAX_number, df_SAX_alphabet, df_SAX_number_diff] = ifeel_feature_transformation(df_wide, alphabet_size,time_business_start,time_business_end)
            print(f"{uid}: Transformation succesful")
            
        except Exception as e:
            print(f"{uid}: Transformation failed — {e}")
            # Append clean placeholder rows with matching structure
            all_global_features.append(pd.DataFrame({"ean_id": [uid], "date": [pd.NaT]}))
            all_peak_features.append(pd.DataFrame({"ean_id": [uid], "date": [pd.NaT]}))
            continue
    
        # Collect global features in a list
        global_features_list = []
        
        for i in range(df_raw.shape[0]):
            ts = df_raw.iloc[i]
            ts_diff = df_raw_diff.iloc[i]
            feature_global_all_each = (
                ifeel_feature_global(ts, ts_diff, sample_interval_in_hour)
                .global_all()
                .T
            )
            # Convert Series to dict so labels become keys
            global_features_list.append(feature_global_all_each.to_dict())
        
        # Create DataFrame directly from list of dicts
        global_features_df = pd.DataFrame(global_features_list)
        global_features_df['ean_id'] = uid
        global_features_df['date'] = df_raw.index
        
        all_global_features.append(global_features_df) 
        print(f"{uid}: Global features extraction succesful")
    
        # Peak feature extraction
        peak_features_list = []
        
        for i in range(df_raw.shape[0]):  # loop over all rows
            ts_sax = df_SAX_number.iloc[i]
            ts_sax_diff = df_SAX_number_diff.iloc[i]
            feature_peak_all_each = (
                ifeel_feature_peak_period(ts_sax, ts_sax_diff, alphabet_size, sample_interval_in_hour)
                .T
            )
        
            # Convert Series to dict so labels become columns
            feature_dict = feature_peak_all_each.to_dict()
            peak_features_list.append(feature_dict)
    
        # Create final DataFrame from list of dicts
        peak_features_df = pd.DataFrame(peak_features_list)
        peak_features_df['ean_id'] = uid
        peak_features_df['date'] = df_SAX_number.index
        
        all_peak_features.append(peak_features_df)
        print(f"{uid}: Peak features extraction succesful")
    
    global_features_all = pd.concat(all_global_features, ignore_index=True)
    peak_features_all = pd.concat(all_peak_features, ignore_index=True)
    
    ###################################################################
    # Merging
    
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
    df = pd.merge(global_features_all, peak_features_all, on=['ean_id', 'date'], how='outer')
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week
    df["moy"] = df["week"].map(week_to_moy)    
    
    # Peak time and peak duration are lists, only maintain the peak time and peak duration for the peak with the longest duration
    df[['peak_time', 'peak_duration']] = df.apply(
        lambda r: pd.Series((lambda i: (r['peak_time'][i], r['peak_duration'][i]))(np.argmax(r['peak_duration']))
                            if isinstance(r['peak_duration'], list) and len(r['peak_duration']) > 0
                            else (np.nan, np.nan)),
        axis=1
    )
    
    feature_cols = [c for c in df.columns if c not in ['ean_id', 'date', 'year', 'moy', 'week']]

    collapsed_year = (
        df.groupby(['ean_id', 'year'])[feature_cols]
        .agg(['min', 'max', 'median', 'mean', 'std'])
        .reset_index()
    )
    
    collapsed_month = (
        df.groupby(['ean_id', 'moy'])[feature_cols]
        .agg(['min', 'max', 'median', 'mean', 'std'])
        .reset_index()
    )
    
    collapsed_week = (
        df.groupby(['ean_id', 'year', 'week'])[feature_cols]
        .agg(['min', 'max', 'median', 'mean', 'std'])
        .reset_index()
    )

    collapsed_year.columns = [
        '_'.join(filter(None, col)).strip() for col in collapsed_year.columns.to_flat_index()
    ]
    
    
    collapsed_month.columns = [
        '_'.join(filter(None, col)).strip() for col in collapsed_month.columns.to_flat_index()
    ]

    collapsed_week.columns = [
        '_'.join(filter(None, col)).strip() for col in collapsed_week.columns.to_flat_index()
    ]
    
    return (collapsed_year, collapsed_month, collapsed_week)
