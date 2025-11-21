# after finishing exploratry data analysis, we found that certain features need to be engineered
# 1-Lap Performance Features
# 2-Weather-Adjusted Features
# 3-Driver/Vehicle Derived Features
# 4-Race Progression Features
# 5-Outlier and Consistency Metrics
# 6-Aggregate / Summary Features (per vehicle)


try:
    import pandas 
    import numpy as np
except Exception as e:
    pass  # Handle the exception or log it if necessary

def engineer_features(df, sort_cols=None):

    if sort_cols is None:
        # sensible default: order by vehicle and lap number (or a timestamp if available)
        sort_cols = ['vehicle_id', 'lap']
    

    # Work on a copy to avoid side effects
    df = df.copy()

    # sort so that diffs and cumsums are meaningful
    df.sort_values(by=sort_cols, inplace=True)

    # Lap-level features
    # mean lap per vehicle
    mean_lap_per_vehicle = df.groupby('vehicle_id')['lap_time_s'].transform('mean')
    df['lap_diff_from_mean'] = df['lap_time_s'] - mean_lap_per_vehicle
    #ratio of lap time to mean lap time
    df['lap_time_ratio'] = df['lap_time_s'] / mean_lap_per_vehicle
    ## lap-to-lap change within each vehicle 
    df['lap_time_change'] = df.groupby('vehicle_id')['lap_time_s'].diff()  # lap-to-lap improvement
    # CUMULATIVE TIME PER VEHICLE
    df['cumulative_time'] = df.groupby('vehicle_id')['lap_time_s'].cumsum()

    # Weather influence metrics
    df['temp_lap_ratio'] = df['lap_time_s'] / df['AIR_TEMP']
    df['wind_lap_ratio'] = df['lap_time_s'] / (df['WIND_SPEED'] + 0.1)  # avoid div by 0
    df['humidity_effect'] = df['HUMIDITY'] * df['lap_time_s']

    # Derived position performance
    df['position_norm'] = df['POSITION'] / df['POSITION'].max()
    df['speed_efficiency'] = df['avg_speed_est'] / df['FL_KPH']
    df['fast_lap_gap'] = df['lap_time_s'] - df['FL_TIME_SEC']

    # Progress through race
    df['lap_progress'] = df['lap'] / df['LAPS'].replace(0, np.nan)  # avoid div by 0
    df['is_first_lap'] = (df['lap'] == 1).astype(int)
    df['is_last_lap'] = (df['lap'] == df['LAPS']).astype(int)

    lap_std = df.groupby('vehicle_id')['lap_time_s'].transform('std')
    df['lap_consistency'] = 1.0 / (1.0 + lap_std)

    # Aggregate / Summary Features per vehicle
        # --- Aggregate / Summary Features per vehicle (robust across pandas versions) ---
    grouped = df.groupby('vehicle_id').agg({
        'lap_time_s': ['mean', 'std', 'min', 'max', 'count'],
        'avg_speed_est': ['mean'],
        'AIR_TEMP': ['mean'],
        'HUMIDITY': ['mean'],
        'WIND_SPEED': ['mean']
    })

    # flatten MultiIndex columns created by the agg
    grouped.columns = ['_'.join(filter(None, col)).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    # rename to the desired names
    rename_map = {
        'lap_time_s_mean': 'mean_lap_time',
        'lap_time_s_std': 'lap_std',
        'lap_time_s_min': 'min_lap',
        'lap_time_s_max': 'max_lap',
        'lap_time_s_count': 'n_laps',
        'avg_speed_est_mean': 'mean_speed',
        'AIR_TEMP_mean': 'mean_air_temp',
        'HUMIDITY_mean': 'mean_humidity',
        'WIND_SPEED_mean': 'mean_wind'
    }
    grouped.rename(columns=rename_map, inplace=True)

    # safe fills for std where n_laps == 1 -> std NaN
    if 'lap_std' in grouped.columns:
        grouped['lap_std'] = grouped['lap_std'].fillna(0.0)

    # merge the aggregated features back to df
    df = df.merge(grouped, on='vehicle_id', how='left')


    

    return df

if __name__ == "__main__":
    df = pandas.read_csv('data/cleaned/cleaned_merged_race_data.csv')
    df_fe = engineer_features(df)

    # Save engineered dataframe
    df_fe.to_csv('data/cleaned/feature_engineered_race_data.csv', index=False)
    print("✅ Feature-engineered dataset saved to data/cleaned/feature_engineered_race_data.csv")# Merge aggregated features back to main dataframe