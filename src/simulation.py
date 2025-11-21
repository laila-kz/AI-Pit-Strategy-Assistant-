#This file simulates a real-time race feed
#Input: Live lap-by-lap telemetry data

#Output: Model predicts probability of pit (should_pit_prob)

# Simulate real-time lap-by-lap telemetry input →
# Use your trained ML model to predict pit stop probability (should_pit_prob) →
# Generate a strategy decision:

# BOX NOW (prob ≥ 0.7)

# PREPARE PIT (0.4 ≤ prob < 0.7)

# STAY OUT (prob < 0.4)

#Steps:
# 1- Import Required Libraries
#2-Load the Trained Model
# 3-Load or Simulate Live Telemetry Data
# 4-Simulate Live Feed + Make Predictions
# 5-Log or Visualize the Output


# 1- Import Required Libraries
import time
import pandas  as pd 
import joblib 
import numpy as np 
import random
import matplotlib.pyplot as plt
from datetime import datetime


# 2-Load the Trained Model
model = joblib.load("models/best_model.pkl")
print( "Model loaded successfully.")

# 3-Load or Simulate Live Telemetry Data
telemetry_data = pd.read_csv("data/cleaned/feature_engineered_race_data.csv")

# a little preview 
print(f"Loaded {len(telemetry_data)} laps of telemetry data.")

# 4-Simulate Live Feed + Make Predictions
    # Loop over each lap (row):

    # Extract features

    # Predict pit probability (should_pit_prob)

    # Apply decision logic

    # Print the action

    # Wait a few seconds (simulate real-time feed)

# Choose one vehicle and take its first up-to-20 laps
if 'vehicle_id' not in telemetry_data.columns:
    print("Warning: 'vehicle_id' column not found. Using first 20 rows of the dataset.")
    telemetry_data = telemetry_data.head(20).reset_index(drop=True)
else:
    vehicle_ids = telemetry_data['vehicle_id'].dropna().unique()
    if len(vehicle_ids) == 0:
        print("Warning: no vehicle_id values found. Using first 20 rows of the dataset.")
        telemetry_data = telemetry_data.head(20).reset_index(drop=True)
    else:
        # pick the first vehicle_id (change to random.choice(vehicle_ids) to pick randomly)
        vehicle_to_sim = vehicle_ids[0]
        print(f"Simulating vehicle_id: {vehicle_to_sim}")

        vehicle_df = telemetry_data[telemetry_data['vehicle_id'] == vehicle_to_sim].copy()

        # If 'lap' exists, keep one row per lap sorted by lap number and take up to 20 laps.
        if 'lap' in vehicle_df.columns:
            vehicle_df = vehicle_df.sort_values('lap').drop_duplicates(subset=['lap']).head(20)
        else:
            # fallback: just take the first 20 rows for that vehicle
            vehicle_df = vehicle_df.head(20)

        telemetry_data = vehicle_df.reset_index(drop=True)
        print(f"Selected {len(telemetry_data)} laps for vehicle {vehicle_to_sim}.")

# Ensure same columns as training
try:
    # Try LightGBM Booster feature names first (most reliable for LGBMClassifier)
    try:
        model_features = model.booster_.feature_name()
    except Exception:
        # Some sklearn wrappers expose feature_name_ after fit
        model_features = getattr(model, 'feature_name_', None)

    if model_features:
        print(f"Model feature names found ({len(model_features)}): {model_features[:10]}{'...' if len(model_features)>10 else ''}")

        # If any feature is missing in telemetry, create a numeric column filled with 0
        missing = [c for c in model_features if c not in telemetry_data.columns]
        if missing:
            print(f"Adding {len(missing)} missing features to telemetry data with zeros: {missing}")
            for c in missing:
                telemetry_data[c] = 0

        # Select model-ordered columns and coerce to numeric (non-convertible -> NaN)
        telemetry_aligned = telemetry_data[model_features].apply(pd.to_numeric, errors='coerce')
    else:
        # Fallback: use numeric-only columns from telemetry data
        print("Model feature names not found; falling back to numeric columns from telemetry data")
        telemetry_aligned = telemetry_data.select_dtypes(include=[float, int]).copy()

    # Drop any completely-non-numeric rows/columns if present and fill NaNs with 0
    telemetry_aligned = telemetry_aligned.fillna(0)
    # Use telemetry_aligned for prediction loop below
    telemetry_for_pred = telemetry_aligned
except Exception as ex:
    print("Failed to align telemetry with model features:", ex)
    raise


result =[]
plt.ion() #interactive mode on
for idx in telemetry_for_pred.index:
    # take a single-row DataFrame (preserves column names) to avoid sklearn "invalid feature names" warning
    features_df = telemetry_for_pred.loc[[idx]].astype(float)
    # Predict probability (model will accept a DataFrame with matching column names)
    try:
        should_pit_prob = model.predict_proba(features_df)[0][1]
    except Exception as e:
        print(f"Prediction failed on index {idx}:", e)
        # fallback: skip this row
        continue

    #decision logic
    if should_pit_prob >= 0.7:
        action = "BOX NOW"
    elif should_pit_prob >= 0.4:
        action = "PREPARE PIT"
    else:
        action = "STAY OUT"
    
    # Use lap number from telemetry if available, else use dataframe index
    lap_display = None
    if 'lap' in telemetry_data.columns:
        try:
            lap_display = int(telemetry_data.loc[idx, 'lap'])
        except Exception:
            lap_display = None
    lap_num = lap_display if lap_display is not None else idx
    result.append({"lap": lap_num, "pit_prob": should_pit_prob, "decision": action})
    lap_label = f"Lap {lap_display}" if lap_display is not None else f"Row {idx}"
    print(f"{lap_label}: Pit Prob = {should_pit_prob:.2f}, Action = {action}")

    plt.clf()
    plt.bar(['STAY OUT', 'PREPARE PIT', 'BOX NOW'],
        [1 if action == x else 0 for x in ['STAY OUT', 'PREPARE PIT', 'BOX NOW']],
        color=['green' if action == 'STAY OUT' else 'orange' if action == 'PREPARE PIT' else 'red'])
    plt.title(f"{lap_label}: Decision - {action}")
    plt.pause(0.5)  # Pause to update the plot

    
    reaction_time = random.uniform(0.1, 0.3)
    time.sleep(reaction_time)  # Wait for a random time to simulate real-time feed
    # This makes your simulation feel like a real pit decision cycle.
plt.ioff()
plt.show()



# Convert to DataFrame
simulation_df = pd.DataFrame(result)

print("\nSimulation Summary:")
print(simulation_df['decision'].value_counts())

simulation_df.to_csv("outputs/simulation_results.csv", index=False)
print("Simulation complete! Results saved to outputs/simulation_results.csv")
#save timestamped version too
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
simulation_df.to_csv(f"outputs/simulation_results_{timestamp}.csv", index=False)
print(" Simulation complete! Results saved to outputs/")



    
