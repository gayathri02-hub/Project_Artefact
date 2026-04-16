import pandas as pd
import numpy as np
import joblib
import time
import json
import os
import math
from tqdm import tqdm
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- DYNAMIC CONFIGURATIONS FOR EXPERIMENT 4 ---
# Read from Docker environment variables, with safe defaults
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 1000)) 
RF_THRESHOLD = float(os.environ.get('RF_THRESHOLD', 0.5)) 

print(f"Starting IoT Edge ML Inference Service...")
print(f"Configuration -> BATCH_SIZE: {BATCH_SIZE}, RF_THRESHOLD: {RF_THRESHOLD}")

# =====================================================================
# 1. LOAD ARTIFACTS
# =====================================================================
try:
    rf_model = joblib.load('rf_v1.pkl')
    ocsvm_model = joblib.load('ocsvm_v1.pkl')
    preprocessor = joblib.load('preprocess_v1.pkl')
    print("Models and preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    exit()

# Extract preprocessor components
scaler = preprocessor['scaler']
encoder = preprocessor['encoder']
selector = preprocessor['selector']

# --- ROBUST BENIGN LABEL IDENTIFICATION ---
benign_classes = [cls for cls in encoder.classes_ if 'benign' in str(cls).lower()]
if len(benign_classes) > 0:
    benign_encoded_value = encoder.transform([benign_classes[0]])[0]
else:
    benign_encoded_value = 0 
    
print(f"Identified Benign encoded value as: {benign_encoded_value}")

# Pre-calculate the index of the benign class for probability checks
try:
    benign_idx = np.where(rf_model.classes_ == benign_encoded_value)[0][0]
except IndexError:
    benign_idx = 0 # Fallback 

# =====================================================================
# 2. STREAMING DATA PIPELINE 
# =====================================================================
# =====================================================================
# 2. STREAMING DATA PIPELINE 
# =====================================================================
print("Opening live traffic stream (test_set.parquet) in memory-safe batches...")

try:
    parquet_file = pq.ParquetFile('test_set.parquet')
    
    # NEW: Calculate exactly how many rows and batches exist for the progress bar
    total_rows = parquet_file.metadata.num_rows
    total_batches = math.ceil(total_rows / BATCH_SIZE)
    
except Exception as e:
    print(f"Error loading parquet file: {e}")
    exit()

rf_binary_predictions = [] 
ocsvm_predictions = []
latencies = []
y_true_encoded = []

batch_count = 0
alert_printed = False

print(f"\nBeginning real-time inference under strict edge memory constraints...")
print(f"Total packets to process: {total_rows} across {total_batches} batches.\n")

# NEW: Wrap iter_batches with tqdm to generate the live progress bar
for batch in tqdm(parquet_file.iter_batches(batch_size=BATCH_SIZE), total=total_batches, desc="Inference Progress", unit="batch"):
    batch_df = batch.to_pandas()
    
    X_live_batch = batch_df.drop(columns=['Label'])
    
    try:
        y_true_batch = encoder.transform(batch_df['Label'])
    except ValueError:
        y_true_batch = [1] * len(batch_df) 
        
    y_true_encoded.extend(y_true_batch)

    # =====================================================================
    # 3. PREPROCESS INCOMING BATCH
    # =====================================================================
    X_live_scaled = scaler.transform(X_live_batch.astype(np.float32))
    X_live_selected = selector.transform(X_live_scaled)

    # =====================================================================
    # 4. RUN DUAL-MODEL INFERENCE ENGINE (WITH THRESHOLDS)
    # =====================================================================
    start_time = time.time()
    
    # Supervised Model (Random Forest) - USING PROBABILITIES NOW
    rf_probs = rf_model.predict_proba(X_live_selected)
    
    # Calculate the probability that the packet is an attack (1.0 - benign probability)
    attack_probs = 1.0 - rf_probs[:, benign_idx]
    
    # Apply the custom threshold (0 = Normal, 1 = Attack)
    rf_pred_binary = (attack_probs >= RF_THRESHOLD).astype(int)
    rf_binary_predictions.extend(rf_pred_binary)
    
    # Anomaly Detection Model (One-Class SVM)
    ocsvm_pred = ocsvm_model.predict(X_live_selected)
    ocsvm_pred_mapped = [0 if p == 1 else 1 for p in ocsvm_pred]
    ocsvm_predictions.extend(ocsvm_pred_mapped)
    
    end_time = time.time()
    
    # Calculate latency per packet in this batch
    latencies.append((end_time - start_time) / len(X_live_selected))
    
    # =====================================================================
    # 5. ALERTING & LOGGING 
    # =====================================================================
    if not alert_printed:
        for idx, (rf_p, ocsvm_p) in enumerate(zip(rf_pred_binary, ocsvm_pred_mapped)):
            if rf_p == 1 or ocsvm_p == 1:
                alert = {
                    "timestamp": time.time(),
                    "alert_type": "MALICIOUS_TRAFFIC" if rf_p == 1 else "ZERO_DAY_ANOMALY",
                    "rf_attack_flag": int(rf_p),
                    "ocsvm_flag": int(ocsvm_p)
                }
                print(f"\n[ALERT GENERATED] --> {json.dumps(alert)}")
                alert_printed = True
                break

    batch_count += 1

print(f"\nSuccessfully processed {batch_count} batches (approx. {batch_count * BATCH_SIZE} packets)!")

# =====================================================================
# 6. PERFORMANCE EVALUATION (Experiment Metrics)
# =====================================================================
avg_latency_ms = np.mean(latencies) * 1000

# Convert true labels to binary (0 = Normal, 1 = Attack)
y_true_binary = [0 if y == benign_encoded_value else 1 for y in y_true_encoded]

print("\n=== FINAL EDGE DEPLOYMENT METRICS ===")
print(f"Average Inference Latency: {avg_latency_ms:.4f} milliseconds per packet")
print(f"Random Forest Accuracy:  {accuracy_score(y_true_binary, rf_binary_predictions):.4f}")
print(f"Random Forest Precision: {precision_score(y_true_binary, rf_binary_predictions):.4f}")
print(f"Random Forest Recall:    {recall_score(y_true_binary, rf_binary_predictions):.4f}")
print(f"Random Forest F1-Score:  {f1_score(y_true_binary, rf_binary_predictions):.4f}")
print("=====================================\n")