import pandas as pd
import numpy as np

path = r'data/cleaned/feature_engineered_race_data.csv'
print('Loading', path)
df = pd.read_csv(path)
print('\nColumns:', df.columns.tolist())

# Quick target check if present
if 'should_pit' in df.columns:
    print('\nTarget should_pit value counts:')
    print(df['should_pit'].value_counts(dropna=False))

# Show top 10 rows
print('\nHead of dataframe:')
print(df.head(10).to_string())

# Check nunique for key columns and engineered names
cols_to_check = ['lap_time_s','avg_speed_est','mean_lap_time','performance_degradation','lap_consistency','fast_lap_gap','speed_efficiency','AIR_TEMP','RAIN','POSITION']
existing = [c for c in cols_to_check if c in df.columns]
print('\nNunique of important columns:')
print(df[existing].nunique())

# Print descriptive stats for columns that model used earlier (if present)
check_cols = [c for c in df.columns if c.lower().startswith(('lap','avg','speed','perf','rain','air','position','humidity','wind','temp'))]
check_cols = sorted(set(check_cols))[:50]
print('\nDescriptive stats for a sample of columns:')
print(df[check_cols].describe().T)

# Check for columns that are all zeros or constant
const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
print('\nConstant columns (nunique<=1):', const_cols)

# If vehicle_id exists, show sample unique values and dtype
if 'vehicle_id' in df.columns:
    print('\nvehicle_id sample values (first 10):', df['vehicle_id'].unique()[:10])
    print('vehicle_id dtype:', df['vehicle_id'].dtype)

print('\nDone')
