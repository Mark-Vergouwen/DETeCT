import os
import yaml
from pathlib import Path
import pandas as pd
from tsfeatures import tsfeatures 

def compute_tsfeatures(df, freq, agg):
    
    if agg == "yearly":
        
        # --- Convert datetime & aggregate hourly ---
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = (
             df.groupby(["EAN_ID", pd.Grouper(key='datetime', freq='H')], as_index=False)
                  .agg({'afname_kwh': 'sum'})
        )
         
        # --- Rename columns for tsfeatures ---
        df = df.rename(columns={'EAN_ID': 'unique_id', 'datetime': 'ds', 'afname_kwh': 'y'})
        all_yearly = []
    
        # --- YEARLY FEATURES ---
        yearly_features_list = []
        
        for uid in df['unique_id'].unique():
            # Create a smaller DataFrame for this household
            sub_df = df[df['unique_id'] == uid].copy()
            feats = tsfeatures(sub_df, freq=freq)
            print(f"Success yearly {uid}")
            feats['unique_id'] = uid
            yearly_features_list.append(feats)
        
         
        if yearly_features_list:
            df_yearly = pd.concat(yearly_features_list, ignore_index=True)
            drop_cols = [
                'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2',
                'series_length', 'seasonal_period', 'nperiods'
            ]
            df_yearly = df_yearly.drop(columns=[c for c in drop_cols if c in df_yearly.columns], errors='ignore')
            df_yearly = df_yearly.sort_values(by=['unique_id']).reset_index(drop=True)
            all_yearly.append(df_yearly)
        else:
            df_yearly = pd.DataFrame()   
        
    elif agg == "monthly":
               
        # Preliminary function definitions
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
              
        # --- Convert datetime & aggregate hourly ---
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = (
            df.groupby(['EAN_ID', pd.Grouper(key='datetime', freq='H')], as_index=False)
                .agg({'afname_kwh': 'sum'})
        )
            
        df = df.rename(columns={'EAN_ID': 'unique_id', 'datetime': 'ds', 'afname_kwh': 'y'})
             
        df['date'] = df['ds'].dt.date
        df['week'] = df['ds'].dt.isocalendar().week.astype(int)
        df["moy"] = df["week"].map(week_to_moy)    
            
        monthly_features_list = []
                
        for uid in df['unique_id'].unique():
                
            for moy in df['moy'].unique():
                    
                # Create a smaller DataFrame for this household
                sub_df = df[(df['unique_id'] == uid) & (df['moy'] == moy)].copy()
                feats = tsfeatures(sub_df[['unique_id', 'ds', 'y']], freq=freq)
                print(f"Success monthly {uid}, month {moy}")
                feats['moy'] = moy
                monthly_features_list.append(feats)
            
        if monthly_features_list:
            df_monthly = pd.concat(monthly_features_list, ignore_index=True)
            drop_cols = [
                'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2',
                'series_length', 'seasonal_period', 'nperiods'
            ]
            df_monthly = df_monthly.drop(columns=[c for c in drop_cols if c in df_monthly.columns], errors='ignore')
            df_monthly = df_monthly.sort_values(by=['unique_id']).reset_index(drop=True)
        else:
            df_monthly = pd.DataFrame()   
       