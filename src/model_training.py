#Train and evaluate the predictive model
# 1 - Load the feature_dataset.csv

# 2 - Split it into training and testing sets.

# 3 - Train your chosen model (e.g., regression if predicting lap time, classification if predicting pit stop timing).

# 4 - Evaluate accuracy (RMSE, R², precision, etc.)

# 5 - Save the model to models/best_model.pkl


#what does the model do :It predicts the optimal moment to call the car in, including human reaction time of the pit crew.
#That means the model must predict whether the next 1–2 laps will be an ideal pit window
#it’s a decision-making model that answers:

#“Should we pit this lap (or within the next N seconds)?”

#meaning it is a classification problem 

#the algorithem that we will use is Gradient Boosting (XGBoost, LightGBM, or CatBoost)
#why: Handles nonlinear relationships, small to medium datasets, high accuracy, easy to explain via feature importance


#import dependencies
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold, GroupShuffleSplit
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.metrics import precision_score, recall_score


# Load the feature dataset
df = pd.read_csv('data/cleaned/feature_engineered_race_data.csv')


# create target variable
#Predict whether the next 1–2 laps represent an ideal pit window (binary classification)

#label each lap as 1 (ideal pit window) or 0 (not ideal)
# i need to create a new column 'ideal_pit_window' in the dataset

# --- Step 1: Sort by vehicle and lap to ensure proper order ---
df = df.sort_values(by=['vehicle_id', 'lap']).reset_index(drop=True)

# --- AGGRESSIVE CLEANING (remove bad rows, handle infs, winsorize) ---
print('\n=== AGGRESSIVE DATA CLEANING ===')
# Replace infinite values with NaN for numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

# Drop rows with too many missing numeric values (>50%)
row_missing_frac = df[num_cols].isna().mean(axis=1)
drop_rows = row_missing_frac > 0.5
if drop_rows.any():
    print(f"Dropping {drop_rows.sum()} rows with >50% missing numeric values")
    df = df.loc[~drop_rows].reset_index(drop=True)

# Remove obvious lap outliers (non-positive or extremely large accidental values)
if 'lap' in df.columns and df['lap'].dtype.kind in 'ifu':
    lap_max = df['lap'].quantile(0.999)
    before = len(df)
    df = df[(df['lap'] > 0) & (df['lap'] <= lap_max)].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with extreme 'lap' values")

# Winsorize numeric columns at 1st/99th percentiles to cap extreme outliers
for c in num_cols:
    if c in df.columns:
        lo = df[c].quantile(0.01)
        hi = df[c].quantile(0.99)
        if pd.notna(lo) and pd.notna(hi) and lo < hi:
            df[c] = df[c].clip(lower=lo, upper=hi)

# Replace remaining infinities and fill NaNs (temporary) with median of column
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
for c in num_cols:
    if c in df.columns:
        if df[c].isna().all():
            # drop wholly-NA numeric columns later when building features
            continue
        med = df[c].median(skipna=True)
        df[c] = df[c].fillna(med)

print('Cleaning complete. Data shape:', df.shape)

# === MORE REALISTIC TARGET CREATION ===
print("\n=== CREATING REALISTIC TARGET ===")

# Use FUTURE performance to predict pit stops (more realistic)
df = df.sort_values(by=['vehicle_id', 'lap']).reset_index(drop=True)

# Look ahead 2 laps to see if performance drops (simulating real prediction)
df['future_slowdown'] = (
    df.groupby('vehicle_id')['avg_speed_est'].shift(-2) < 
    df.groupby('vehicle_id')['avg_speed_est'].rolling(5, min_periods=1).mean().reset_index(0, drop=True) * 0.97
)

# Look ahead for position loss
df['future_position_loss'] = df.groupby('vehicle_id')['POSITION'].diff(-2) > 0.5

# Combine signals - predict pit stops 2 laps in advance
df['should_pit'] = (
    (df['future_slowdown'].fillna(False)) |
    (df['future_position_loss'].fillna(False)) |
    (df['RAIN'] > 0.2)  # Heavy rain
).astype(int)

# Remove last 2 laps (no future data)
df = df[df['lap'] <= df['lap'].max() - 2].reset_index(drop=True)

# Remove first 10 laps (unlikely strategic pits)
df['should_pit'] = np.where(df['lap'] <= 10, 0, df['should_pit'])

print("Realistic target distribution:")
dist = df['should_pit'].value_counts(normalize=True)
print(dist)

# If the realistic lookahead target produced a single class (no positives),
# fall back to a more permissive, legacy target so we can continue development.
if df['should_pit'].nunique() <= 1:
    print("WARNING: lookahead target is single-class; falling back to a legacy performance-based target to avoid degenerate training set")
    # Recompute rolling lap time and performance_degradation
    df['rolling_lap_time'] = df.groupby('vehicle_id')['lap_time_s'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['performance_degradation'] = ((df['lap_time_s'] / df['rolling_lap_time'] - 1) * 100)
    df['temp_diff'] = df.groupby('vehicle_id')['AIR_TEMP'].diff()
    df['should_pit'] = np.where(
        (df['performance_degradation'] > 3.0) | (df['RAIN'] > 0) | (abs(df['temp_diff']) > df['temp_diff'].std() * 2),
        1, 0
    )
    print("Fallback target distribution:")
    print(df['should_pit'].value_counts(normalize=True))


# === SAFE FEATURE SET - NO ENGINEERED METRICS ===
print("\n=== USING ONLY BASIC REAL-TIME FEATURES ===")

basic_safe_features = [
    # Raw performance metrics (no ratios, no consistency measures)
    'lap_time_s', 'avg_speed_est', 
    
    # Race context
    'lap', 'POSITION', 'interval_s',
    
    # Raw weather data (no engineered effects)
    'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'RAIN',
    
    # Basic position trends (if available)
    'position_norm', 'lap_progress'
]

# Filter to only existing columns
basic_safe_features = [f for f in basic_safe_features if f in df.columns]
print(f"Using {len(basic_safe_features)} basic features: {basic_safe_features}")

X = df[basic_safe_features].copy()

# Add simple rolling averages (3-lap window)
for col in ['lap_time_s', 'avg_speed_est', 'POSITION']:
    if col in X.columns:
        X[f'{col}_rolling_3'] = df.groupby('vehicle_id')[col].rolling(3, min_periods=1).mean().reset_index(0, drop=True)

# Fill missing values (forward/backward then 0)
X = X.ffill().bfill().fillna(0)  # Using newer pandas methods

print(f"Final feature set: {X.shape[1]} features")
print("Feature summary:")
print(X.describe().T[['mean', 'std', 'min', 'max']])

# Define target variable
y = df['should_pit']

# === CRITICAL DATA LEAKAGE DETECTION ===
print("\n" + "="*60)
print("[!] DATA LEAKAGE DETECTION")
print("="*60)

leaky_features_found = []
for feature in X.columns:
    try:
        # compute correlation with target (robust to constant features)
        f = pd.to_numeric(X[feature], errors='coerce')
        # align indices
        common_idx = f.index.intersection(y.index)
        corr = 0.0
        if f.loc[common_idx].nunique() > 1:
            corr = abs(np.corrcoef(f.loc[common_idx].fillna(0), y.loc[common_idx].fillna(0))[0,1])
        # check perfect mapping
        temp_df = pd.DataFrame({'feature': f, 'target': y})
        perfect_mapping = False
        try:
            perfect_mapping = temp_df.groupby('feature')['target'].nunique().max() == 1
        except Exception:
            perfect_mapping = False

        if corr > 0.9 or perfect_mapping:
            leaky_features_found.append(feature)
            print(f"[!] LEAKY FEATURE: {feature}")
            print(f"   Correlation with target: {corr:.3f}")
            print(f"   Perfect mapping: {perfect_mapping}")
    except Exception as e:
        print('Leakage check failed for', feature, e)

if leaky_features_found:
    print(f"\n[!] REMOVING LEAKY FEATURES: {leaky_features_found}")
    X = X.drop(columns=leaky_features_found, errors='ignore')
    print(f"Now using {X.shape[1]} safe features")

print("="*60)

# --- CLEANUP: remove identifier columns and constant features that provide no signal ---
# Drop vehicle_id (identifier) to avoid leakage and meaningless numeric encoding
if 'vehicle_id' in X.columns:
    print("Dropping identifier column 'vehicle_id' to avoid leakage")
    X = X.drop(columns=['vehicle_id'])

# Convert all columns to numeric where possible and coerce errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Replace any remaining NaNs (from coercion) with 0
# Replace infinities with NaN so we can detect and handle them
X = X.replace([np.inf, -np.inf], np.nan)

# Drop columns with too many missing / infinite values (>20%)
missing_frac = X.isna().mean()
cols_to_drop = missing_frac[missing_frac > 0.20].index.tolist()
if cols_to_drop:
    print("Dropping columns with >20% missing/inf values:", cols_to_drop)
    X = X.drop(columns=cols_to_drop)

# Impute remaining NaNs with median to preserve distributions
for c in X.columns:
    if X[c].isna().any():
        med = X[c].median()
        X[c] = X[c].fillna(med)

# Detect constant (zero-variance) features and drop them
nunique = X.nunique(dropna=True)
constant_features = nunique[nunique <= 1].index.tolist()
if constant_features:
    print("Dropping constant features (no variance):", constant_features)
    X = X.drop(columns=constant_features)

# Report feature stats for debugging
print("\nFeature Statistics:")
print(X.describe().T[['mean','std','min','max']])
print("\nFeatures with all zeros:")
zero_features = X.columns[(X == 0).all()].tolist()
print(zero_features)

print("\nEngineered features:", X.columns.tolist())
print(f"\nFeatures after interactions: {X.shape[1]}")
print("Added feature interactions and polynomial terms")

# --- DIAGNOSTICS: quick checks to see if any feature carries signal ---
try:
    print("\n--- Pre-balancing diagnostics ---")
    print("X shape:", X.shape)
    print("Target distribution (should_pit):")
    print(df['should_pit'].value_counts(dropna=False))

    # show number of unique values and missing counts per feature
    nunique = X.nunique(dropna=False).sort_values()
    miss = X.isna().sum().sort_values()
    print('\nFeature nunique (small->large):')
    print(nunique.head(20).to_string())
    print('\nTop features with missing values:')
    print(miss[miss>0].head(20).to_string())

    # compute simple mutual information / correlation with target if possible
    from sklearn.feature_selection import mutual_info_classif
    # Need to coerce to numeric copy for MI calculation; fill remaining NaNs with median temporarily
    X_mi = X.copy()
    for c in X_mi.columns:
        if X_mi[c].isna().any():
            X_mi[c] = X_mi[c].fillna(X_mi[c].median())
    try:
        mi = mutual_info_classif(X_mi, df['should_pit'].loc[X_mi.index], discrete_features='auto', random_state=42)
        mi_s = pd.Series(mi, index=X_mi.columns).sort_values(ascending=False)
        print('\nTop 10 features by mutual information with target:')
        print(mi_s.head(10).to_string())
    except Exception as e:
        print('Mutual information calculation failed:', e)
except Exception as e:
    print('Pre-balancing diagnostics failed:', e)

# Define target variable
y = df['should_pit']

# Ensure balanced dataset through undersampling
from sklearn.utils import resample

# Get indices of each class
pit_indices = y[y == 1].index
no_pit_indices = y[y == 0].index

# Undersample the majority class
if len(pit_indices) < len(no_pit_indices):
    no_pit_indices = resample(no_pit_indices, 
                            n_samples=len(pit_indices),
                            random_state=42)
else:
    pit_indices = resample(pit_indices,
                          n_samples=len(no_pit_indices),
                          random_state=42)

# Combine balanced indices
balanced_indices = np.concatenate([pit_indices, no_pit_indices])
X = X.loc[balanced_indices]
y = y.loc[balanced_indices]

print("\nAfter balancing:")
print(y.value_counts(normalize=True))

# Add minimal noise only to continuous features
print("\nAdding minimal noise to continuous features...")
noise_scale = 0.005  # 0.5% noise
for col in X.columns:
    if col not in ['lap', 'vehicle_id'] and X[col].nunique() > 10:  # Only add noise to continuous features
        X[col] = X[col] + np.random.normal(0, noise_scale * X[col].std(), size=len(X))

# Split into training, validation, and testing sets using group-aware splits (vehicle_id)
# Use the original df indices to get vehicle groups aligned with X (we used balanced_indices earlier)
try:
    groups = df.loc[X.index, 'vehicle_id']
except Exception:
    # If vehicle_id alignment fails, fall back to no-group split (but this is not ideal)
    print("WARNING: Could not align vehicle_id for group splits. Falling back to random split.")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
else:
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_temp, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_temp, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Now split X_temp into train/val by groups (preserve groups)
    groups_temp = groups.iloc[train_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx2, val_idx2 = next(gss2.split(X_temp, y_temp, groups=groups_temp))
    X_train, X_val = X_temp.iloc[train_idx2], X_temp.iloc[val_idx2]
    y_train, y_val = y_temp.iloc[train_idx2], y_temp.iloc[val_idx2]

#train the model using LightGBM
# model = lgb.LGBMClassifier(
#     objective='binary',
#     boosting_type='gbdt',
#     n_estimators=100,
#     learning_rate=0.1,
#     random_state=42
# )

# --- BASELINE CHECK: Dummy classifier to verify signal exists ---
from sklearn.dummy import DummyClassifier
print("\n--- Baseline DummyClassifier (most_frequent) ---")
dummy = DummyClassifier(strategy='most_frequent')
try:
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    print(f"Dummy Accuracy: {accuracy_score(y_test, y_dummy):.4f}")
    # Dummy ROC-AUC may not be available if predict_proba isn't supported for strategy
    if hasattr(dummy, 'predict_proba'):
        try:
            print(f"Dummy ROC-AUC: {roc_auc_score(y_test, dummy.predict_proba(X_test)[:,1]):.4f}")
        except Exception:
            pass
except Exception as e:
    print('Baseline dummy failed:', e)


# Define a single well-tuned model with modified parameters for better feature detection
model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    # Moderately relaxed but still regularized model
    learning_rate=0.05,
    n_estimators=300,
    max_depth=5,
    num_leaves=31,
    min_child_samples=20,
    min_child_weight=1e-3,
    min_split_gain=0.0,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    feature_fraction=0.8
)

# Print feature value ranges before training
print("\nFeature value ranges:")
for col in X_train.columns:
    print(f"{col}: {X_train[col].min():.3f} to {X_train[col].max():.3f}")

# Train with early stopping and feature importance monitoring
print("\nTraining model with early stopping...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric=['auc', 'binary_logloss'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ],
    feature_name=list(X_train.columns),
        # Do not explicitly pass vehicle_id as a categorical feature here (we dropped it earlier)
)




# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
y_pred_proba = model.predict_proba(X_test)[:,1]
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# ADD THIS SECTION AFTER model.fit()
print("\n=== OVERFITTING DIAGNOSTICS ===")
# Check train vs test performance
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Overfitting Gap: {train_accuracy - test_accuracy:.4f}")

if train_accuracy - test_accuracy > 0.1:
    print(" HIGH OVERFITTING DETECTED!")
elif train_accuracy - test_accuracy > 0.05:
    print("  Moderate overfitting detected")
else:
    print(" Good generalization")


# cross-validation with multiple metrics
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = {
    'accuracy': [],
    'roc_auc': [],
    'precision': [],
    'recall': []
}

print("\n=== Cross-Validation Results ===")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train on this fold
    model.fit(X_fold_train, y_fold_train)
    
    # Get predictions
    y_fold_pred = model.predict(X_fold_val)
    y_fold_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    
    # Calculate metrics
    cv_metrics['accuracy'].append(accuracy_score(y_fold_val, y_fold_pred))
    cv_metrics['roc_auc'].append(roc_auc_score(y_fold_val, y_fold_pred_proba))
    cv_metrics['precision'].append(accuracy_score(y_fold_val, y_fold_pred))
    cv_metrics['recall'].append(accuracy_score(y_fold_val, y_fold_pred))
    
    print(f"\nFold {fold} Results:")
    print(f"Accuracy: {cv_metrics['accuracy'][-1]:.4f}")
    print(f"ROC-AUC: {cv_metrics['roc_auc'][-1]:.4f}")
    
print("\nOverall Cross-Validation Results:")
for metric, scores in cv_metrics.items():
    print(f"{metric.upper()}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

#evaluate model on test set
#confusion matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.colorbar()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='--', color='gray', linewidth=1, label='Random chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(recall, precision, label=f'AP = {average_precision_score(y_test, y_pred_proba):.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Feature importance visualization
print("\nFeature Importance Analysis:")
feature_names = X_train.columns
if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    importances[:20].plot(kind='bar')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Most Important Features:")
    print(importances[:10].to_string())


# === SANITY CHECK ===
print("\n" + "="*50)
print("SANITY CHECK - Should NOT see perfect scores!")
print("="*50)

if accuracy_score(y_test, y_pred) > 0.95:
    print(" CRITICAL: Still getting near-perfect accuracy!")
    print("This indicates persistent data leakage.")
elif accuracy_score(y_test, y_pred) > 0.8:
    print("  High accuracy - may still have some leakage")
else:
    print(" Realistic performance achieved")

# Check if any single feature can predict the target
print("\nSingle-feature predictive power:")
for feature in X.columns[:5]:  # Check first 5 features
    try:
        single_feature_model = lgb.LGBMClassifier(max_depth=3, n_estimators=50, random_state=42)
        single_feature_model.fit(X_train[[feature]], y_train)
        single_score = accuracy_score(y_test, single_feature_model.predict(X_test[[feature]]))
        print(f"  {feature}: {single_score:.3f}")
        if single_score > 0.9:
            print(f"    🚨 {feature} alone can predict the target!")
    except Exception as e:
        print('Single-feature check failed for', feature, e)

#finally save the model
joblib.dump(model, 'models/best_model.pkl')
print("Model saved to models/best_model.pkl")


#we can see some overfitting issues
#The model is achieving near-perfect scores (1.0 accuracy in cross-validation) 
#which is a strong indicator of overfitting, especially given
#potentiial data leakage 
#class imbalance
#feature engineering issues
#hyperparameter tuning


#---- needing some values calculated here in the dashboard -----
print("\n=== SAVING MODEL EVALUATION METRICS TO EXCEL ===")

#compute the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall = recall_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

#the confusion metrics
cm_df= pd.DataFrame(cm,index=['Actual_NoPit(0)', 'Actual_Pit(1)'],
                    columns=['Pred_NoPit(0)', 'Pred_Pit(1)'])

#train/test performance
train_test_perf = pd.DataFrame({
    'Metric': ['Train Acccuracy ', 'Test Accuracy', 'Overfitting Gap'],
    'Value': [train_accuracy , test_accuracy, train_accuracy - test_accuracy]
})

#cross validation
cv_summary = pd.DataFrame({
    'Metric': list(cv_metrics.keys()),
    'Mean': [np.mean(v) for v in cv_metrics.values()],
    'StdDev': [np.std(v) for v in cv_metrics.values()]
})

#combien all the metrics in one sheet
main_metrics =pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'ROC-AUC'],
    'Value': [accuracy, precision, recall, roc_auc]
})

excel_path ='reports/model_evaluation_metrics.xlsx'
with pd.ExcelWriter(excel_path,engine='openpyxl') as writer:
    main_metrics.to_excel(writer, sheet_name='Main_Metrics', index=False)
    train_test_perf.to_excel(writer, sheet_name='Train_Test_Performance', index=False)
    cv_summary.to_excel(writer, sheet_name='Cross_Validation', index=False)
    cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
    # Optional: Save feature importances
    if hasattr(model, 'feature_importances_'):
        importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        importances_df.to_excel(writer, sheet_name='Feature_Importances', index=False)

print(f"✅ All evaluation metrics saved to {excel_path}")