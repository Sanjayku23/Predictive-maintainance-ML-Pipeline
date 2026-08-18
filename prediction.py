import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# -------------------------------------------------------------------------
# 1. SIMULATING UCI PREDICTIVE MAINTENANCE DATASETS 
# (Matches parameters specified in Sanjay's proposal)
# -------------------------------------------------------------------------
np.random.seed(42)
num_samples = 5000  # sensor data

air_temp = np.random.normal(300, 2, num_samples) # Kelvin
process_temp = air_temp + np.random.normal(10, 1, num_samples) # Kelvin
rot_speed = np.random.normal(1500, 150, num_samples) # RPM
torque = np.random.normal(40, 10, num_samples) # Nm
tool_wear = np.random.uniform(0, 240, num_samples) # Minutes

# Initializing all targets as healthy (0: No Failure)
failure_type = np.zeros(num_samples, dtype=int)

# Injecting logical mechanical failure modes matching the proposal rules
for i in range(num_samples):
    # Heat Dissipation Failure (Class 1)
    if (process_temp[i] - air_temp[i] < 8.5) and (rot_speed[i] < 1350):
        failure_type[i] = 1
    # Overstrain Failure (Class 2)
    elif (torque[i] * tool_wear[i] > 11000):
        failure_type[i] = 2
    # Tool Wear Failure (Class 3)
    elif (tool_wear[i] > 220) and (torque[i] > 55):
        failure_type[i] = 3

# Create DataFrame
df = pd.DataFrame({
    'Air_Temp': air_temp,
    'Process_Temp': process_temp,
    'Rotational_Speed': rot_speed,
    'Torque': torque,
    'Tool_Wear': tool_wear,
    'Failure_Type': failure_type
})

# -------------------------------------------------------------------------
# 2. DATA PREPROCESSING
# -------------------------------------------------------------------------
X = df.drop(columns=['Failure_Type'])
y = df['Failure_Type']

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------------
# 3. MODEL TRAINING & EVALUATION
# -------------------------------------------------------------------------
target_names = ['No Failure', 'Heat Dissipation', 'Overstrain', 'Tool Wear']

# --- Model A: Logistic Regression (Baseline) ---


lr_model = LogisticRegression(max_iter=1000, random_state=42)


#lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

print("==================================================")
print(" LOGISTIC REGRESSION (BASELINE) REPORT")
print("==================================================")
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
#print(classification_report(y_test, lr_preds, target_names=target_names, zero_division=0))

print(classification_report(y_test, lr_preds, labels=[0, 1, 2, 3], target_names=target_names, zero_division=0))

# --- Model B: Random Forest Classifier ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # Tree models handle unscaled features flawlessly
rf_preds = rf_model.predict(X_test)

print("==================================================")
print(" RANDOM FOREST CLASSIFIER REPORT")
print("==================================================")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
#print(classification_report(y_test, rf_preds, target_names=target_names, zero_division=0))

print(classification_report(y_test, rf_preds, labels=[0, 1, 2, 3], target_names=target_names, zero_division=0))

# --- Model C: XGBoost Gradient Boosting Classifier ---
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("==================================================")
print(" XGBOOST GRADIENT BOOSTING REPORT")
print("==================================================")
print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
#print(classification_report(y_test, xgb_preds, target_names=target_names, zero_division=0))

print(classification_report(y_test, xgb_preds, labels=[0, 1, 2, 3], target_names=target_names, zero_division=0))
# -------------------------------------------------------------------------
# 4. FEATURE IMPORTANCE ANALYSIS
# -------------------------------------------------------------------------
print("==================================================")
print(" FEATURE IMPORTANCE ANALYSIS (XGBoost)")
print("==================================================")
importances = xgb_model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"Feature: {feature:<20} Importance Score: {importance:.4f}")

# Create new hypothetical sensor readings (1 row or multiple rows)
new_sensor_data = pd.DataFrame([{
    'Air_Temp': 300.5,
    'Process_Temp': 308.2,
    'Rotational_Speed': 1300,  # Low rotational speed
    'Torque': 45.0,
    'Tool_Wear': 210
}])

# 1. Predict using XGBoost (Tree models don't require scaled data)
xgb_prediction = xgb_model.predict(new_sensor_data)
predicted_class = target_names[xgb_prediction[0]]

print(f"Predicted Machine Status (XGBoost): {predicted_class}")

# 2. Predict using Logistic Regression (Requires feature scaling first)
new_sensor_scaled = scaler.transform(new_sensor_data)
lr_prediction = lr_model.predict(new_sensor_scaled)
print(f"Predicted Machine Status (Logistic Regression): {target_names[lr_prediction[0]]}")

import time

def monitor_machine_live(model, target_names, alert_threshold=0.50):
    """
    Simulates a real-time IoT monitoring system checking live machine sensors.
    Triggers an emergency alert if failure probability exceeds the threshold.
    """
    # Simulated live sensor feed arriving minute-by-minute
    # Replace this list with real data pulled from an IoT database or API in production
    live_sensor_stream = [
        {'Air_Temp': 300.0, 'Process_Temp': 310.0, 'Rotational_Speed': 1500, 'Torque': 35.0, 'Tool_Wear': 50},   # Healthy
        {'Air_Temp': 300.2, 'Process_Temp': 310.5, 'Rotational_Speed': 1490, 'Torque': 30.0, 'Tool_Wear': 60},  # Healthy
        {'Air_Temp': 302.0, 'Process_Temp': 311.0, 'Rotational_Speed': 1450, 'Torque': 200.0, 'Tool_Wear': 500},  # Imminent Failure!
    ]

    for reading_num, sensor_reading in enumerate(live_sensor_stream, start=1):
        # Convert incoming single reading to a DataFrame
        input_data = pd.DataFrame([sensor_reading])
        
        # 1. Calculate probability distribution for all failure types
        probabilities = model.predict_proba(input_data)[0]
        
        # 2. Extract risk metrics
        healthy_probability = probabilities[0]
        failure_risk = 1.0 - healthy_probability  # Total risk of any failure
        predicted_class_id = np.argmax(probabilities)
        predicted_failure_name = target_names[predicted_class_id]
        
        print(f"\n--- [Reading {reading_num}] Telemetry Received ---")
        
        # 3. Decision rule: Trigger alert if total risk > threshold
        if failure_risk >= alert_threshold:
            print("🚨 EMERGENCY ALERT: IMMINENT FAILURE DETECTED!")
            print(f"   Predicted Issue:   {predicted_failure_name}")
            print(f"   Risk Probability:  {failure_risk * 100:.1f}%")
            print("   Recommended Action: SHUT DOWN MACHINE IMMEDIATELY FOR INSPECTION.")
        else:
            print(f"✅ Machine Operating Normally (Failure Risk: {failure_risk * 100:.1f}%)")
            
        time.sleep(1) # Simulates time interval between sensor readings

# Execute the live monitoring loop using your trained XGBoost model
monitor_machine_live(xgb_model, target_names, alert_threshold=0.50)