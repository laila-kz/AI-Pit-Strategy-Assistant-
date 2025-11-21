import joblib, pandas as pd, sys

print('Loading model...')
model = joblib.load('models/best_model.pkl')
print('Loading telemetry...')
df = pd.read_csv('data/cleaned/feature_engineered_race_data.csv')
try:
    try:
        mf = model.booster_.feature_name()
    except Exception:
        mf = getattr(model, 'feature_name_', None)
    print('model features found:', bool(mf))
    if mf:
        missing = [c for c in mf if c not in df.columns]
        print('missing count:', len(missing))
        if missing:
            print('example missing:', missing[:10])
            for c in missing:
                df[c] = 0
        features_df = df.loc[[0], mf].apply(pd.to_numeric, errors='coerce').fillna(0)
    else:
        features_df = df.loc[[0]].select_dtypes(include=[float, int]).apply(pd.to_numeric, errors='coerce').fillna(0)
    print('features shape:', features_df.shape)
    print('predicting...')
    print('predict_proba =>', model.predict_proba(features_df)[0])
except Exception as e:
    print('ERROR during prediction test:', e)
    sys.exit(1)
print('Headless predict test completed')
