import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

CSV = r"data/cleaned/feature_engineered_race_data.csv"

def make_lookahead_target(df):
    # copy the same logic used in model_training: future 2-lap slowdown and position loss
    df = df.copy()
    # avoid errors if columns missing
    if 'avg_speed_est' not in df.columns or 'POSITION' not in df.columns:
        raise RuntimeError('required columns missing in CSV')

    df['future_avg_speed_2'] = df['avg_speed_est'].shift(-2)
    df['past_avg_speed_5'] = df['avg_speed_est'].rolling(5, min_periods=1).mean()
    df['future_speed_ratio'] = df['future_avg_speed_2'] / (df['past_avg_speed_5'] + 1e-9)
    df['future_slowdown'] = df['future_speed_ratio'] < 0.97

    df['position_future_2'] = df['POSITION'].shift(-2)
    df['future_position_loss'] = (df['position_future_2'] - df['POSITION']) > 0.5

    df['should_pit_lookahead'] = (df['future_slowdown'] & df['future_position_loss']).astype(int)
    return df


def make_fallback_target(df):
    # simplistic fallback based on lap_time_ratio/performance_degradation if present
    df = df.copy()
    if 'lap_time_s' in df.columns:
        df['lap_time_ratio'] = df['lap_time_s'] / (df['lap_time_s'].rolling(5, min_periods=1).mean() + 1e-9)
        df['should_pit_fallback'] = (df['lap_time_ratio'] > 1.05).astype(int)
    else:
        df['should_pit_fallback'] = 0
    return df


basic_safe = [
    'lap_time_s', 'avg_speed_est', 'lap', 'POSITION',
    'AIR_TEMP', 'HUMIDITY', 'WIND_SPEED', 'RAIN', 'position_norm', 'lap_progress'
]
# also consider rolling_3 versions if present
rolling3 = [c + '_rolling_3' for c in ['lap_time_s', 'avg_speed_est', 'POSITION']]
all_safe = basic_safe + [c for c in rolling3 if c not in basic_safe]


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows, {len(df.columns)} cols\n")

    df = make_lookahead_target(df)
    dist = df['should_pit_lookahead'].value_counts(dropna=False)
    print('Lookahead target distribution:')
    print(dist)
    print()

    if dist.shape[0] == 1:
        print('Lookahead target is single-class; computing fallback target instead.\n')
        df = make_fallback_target(df)
        dist2 = df['should_pit_fallback'].value_counts(dropna=False)
        print('Fallback target distribution:')
        print(dist2)
        target_col = 'should_pit_fallback'
    else:
        target_col = 'should_pit_lookahead'

    # pick safe features that actually exist and coerce to numeric
    features = [c for c in all_safe if c in df.columns]
    print('\nUsing safe features (found):', features)

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df[target_col].fillna(0).astype(int)

    # mutual information
    try:
        mi = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
        mi_series = pd.Series(mi, index=features).sort_values(ascending=False)
        print('\nMutual information (top features):')
        print(mi_series.head(10))
    except Exception as e:
        print('\nCould not compute mutual information:', e)

    print('\nPer-feature basic stats:')
    print(X.describe().loc[['min','mean','std','max']].T)

    # print a quick correlation table to target (pearson) - may show NaN for constants
    corrs = {}
    for f in features:
        try:
            corrs[f] = float(np.corrcoef(X[f], y)[0,1])
        except Exception:
            corrs[f] = np.nan
    corrs_s = pd.Series(corrs).sort_values(key=lambda s: s.abs(), ascending=False)
    print('\nPearson correlation with target (abs desc):')
    print(corrs_s)

if __name__ == '__main__':
    main()
