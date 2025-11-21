import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

p='data/cleaned/feature_engineered_race_data.csv'
df = pd.read_csv(p)
print('DF shape:', df.shape)
print('Columns:', list(df.columns))

# sort
if 'vehicle_id' in df.columns and 'lap' in df.columns:
    df = df.sort_values(['vehicle_id','lap']).reset_index(drop=True)

# quick clean same as script
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
for c in num_cols:
    if df[c].isna().all():
        continue
    med = df[c].median(skipna=True)
    df[c] = df[c].fillna(med)

# compute performance_degradation
if 'lap_time_s' in df.columns and 'vehicle_id' in df.columns:
    df['rolling_lap_time'] = df.groupby('vehicle_id')['lap_time_s'].rolling(window=3, min_periods=1).mean().reset_index(0,drop=True)
    df['performance_degradation'] = ((df['lap_time_s'] / df['rolling_lap_time'] - 1) * 100)
    print('\nperformance_degradation stats:')
    print(df['performance_degradation'].describe())
else:
    print('Missing lap_time_s or vehicle_id, skipping performance_degradation')

# how many rows flagged by the rule in model
if 'performance_degradation' in df.columns and 'AIR_TEMP' in df.columns and 'RAIN' in df.columns:
    temp_diff = df.groupby('vehicle_id')['AIR_TEMP'].diff()
    cfg = ( (df['performance_degradation']>3.0) | (df['RAIN']>0) | (abs(temp_diff) > temp_diff.std()*2) )
    print('\nCount of should_pit by this rule:', cfg.sum(), 'of', len(cfg), f'({cfg.mean():.3f})')

# inspect columns used as features in model script
feat_cols = ['performance_degradation','lap_consistency','fast_lap_gap','avg_speed_est','speed_efficiency','AIR_TEMP','HUMIDITY','WIND_SPEED','RAIN','lap','lap_progress','position_norm','POSITION','humidity_effect','wind_lap_ratio']
print('\nSelected feature columns present and nunique:')
for c in feat_cols:
    if c in df.columns:
        print(c, 'nunique=', df[c].nunique(), 'nans=', df[c].isna().sum(), 'min,max=', df[c].min(), df[c].max())
    else:
        print(c, 'MISSING')

# Build a small X just like the script but minimal
X = pd.DataFrame({
    'recent_performance': df.groupby('vehicle_id')['performance_degradation'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True) if 'performance_degradation' in df.columns else np.nan,
    'performance_trend': df.groupby('vehicle_id')['performance_degradation'].diff() if 'performance_degradation' in df.columns else np.nan,
    'lap_consistency': df['lap_consistency'] if 'lap_consistency' in df.columns else np.nan,
    'fast_lap_gap': df['fast_lap_gap'] if 'fast_lap_gap' in df.columns else np.nan,
    'speed_consistency': df.groupby('vehicle_id')['avg_speed_est'].rolling(window=3, min_periods=1).std().reset_index(0, drop=True) if 'avg_speed_est' in df.columns else np.nan,
    'speed_trend': df.groupby('vehicle_id')['avg_speed_est'].pct_change() if 'avg_speed_est' in df.columns else np.nan,
    'speed_efficiency': df['speed_efficiency'] if 'speed_efficiency' in df.columns else np.nan,
    'temp_change_rate': df.groupby('vehicle_id')['AIR_TEMP'].diff() if 'AIR_TEMP' in df.columns else np.nan,
    'rain_intensity': df['RAIN'] if 'RAIN' in df.columns else np.nan,
    'humidity_effect': df['humidity_effect'] if 'humidity_effect' in df.columns else np.nan,
    'wind_lap_ratio': df['wind_lap_ratio'] if 'wind_lap_ratio' in df.columns else np.nan,
    'lap': df['lap'] if 'lap' in df.columns else np.nan,
    'lap_progress': df['lap_progress'] if 'lap_progress' in df.columns else np.nan,
    'position_norm': df['position_norm'] if 'position_norm' in df.columns else np.nan,
    'POSITION': df['POSITION'] if 'POSITION' in df.columns else np.nan,
    'WIND_SPEED': df['WIND_SPEED'] if 'WIND_SPEED' in df.columns else np.nan,
    'HUMIDITY': df['HUMIDITY'] if 'HUMIDITY' in df.columns else np.nan,
    'AIR_TEMP': df['AIR_TEMP'] if 'AIR_TEMP' in df.columns else np.nan,
})

print('\nBuilt minimal X shape:', X.shape)
print('X nunique (head 30):')
print(X.nunique().sort_values().head(30))
print('Columns with all constant values:')
print([c for c in X.columns if X[c].nunique(dropna=True)<=1])

# mutual info with should_pit
if 'should_pit' in df.columns:
    y = df['should_pit']
else:
    # recreate simplistic rule
    temp_diff = df.groupby('vehicle_id')['AIR_TEMP'].diff()
    y = ((df['performance_degradation']>3.0) | (df['RAIN']>0) | (abs(temp_diff) > temp_diff.std()*2)).astype(int)

X_mi = X.copy()
for c in X_mi.columns:
    if X_mi[c].isna().any():
        X_mi[c] = X_mi[c].fillna(X_mi[c].median())
try:
    mi = mutual_info_classif(X_mi, y, random_state=42)
    mi_s = pd.Series(mi, index=X_mi.columns).sort_values(ascending=False)
    print('\nMutual information (top 10):')
    print(mi_s.head(10).to_string())
except Exception as e:
    print('MI failed:', e)

print('\nSample rows:')
print(pd.concat([X.head(5), y.head(5)], axis=1))

# Quick model sanity check: can a simple classifier pick up signal?
print('\n--- Quick model sanity checks ---')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

X_small = X.copy()
for c in X_small.columns:
    if X_small[c].isna().any():
        X_small[c] = X_small[c].fillna(X_small[c].median())

X_tr, X_te, y_tr, y_te = train_test_split(X_small, y, test_size=0.3, random_state=42, stratify=y)

try:
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr, y_tr)
    p = lr.predict_proba(X_te)[:,1]
    print('Logistic AUC:', roc_auc_score(y_te, p), 'Acc:', accuracy_score(y_te, lr.predict(X_te)))
except Exception as e:
    print('Logistic failed:', e)

try:
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_tr, y_tr)
    p = rf.predict_proba(X_te)[:,1]
    print('RandomForest AUC:', roc_auc_score(y_te, p), 'Acc:', accuracy_score(y_te, rf.predict(X_te)))
    importances = pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    print('\nRF top features:\n', importances.head(10).to_string())
except Exception as e:
    print('RandomForest failed:', e)
