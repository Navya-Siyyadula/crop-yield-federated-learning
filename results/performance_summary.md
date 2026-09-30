# Performance, Results, and Documentation Summary

## 1. Experimental Setup

- **Host Environment:** Windows 11 (AMD64), Intel(R) Core(TM) Processor (4 logical cores), 16 GB System RAM.
- **Software Stack:** Python `3.14.7` (3.12+ compatible), TensorFlow `2.21.0` / Keras 3 (PyTorch backend active in local env), Flower `1.38.0`, cryptography `50.0.1`.
- **Dataset & Frozen Splits:** Preprocessed tabular dataset (`data/processed/`) with 38 input features and target `yield_kg_per_hectare`.
  - Train split: 350 samples
  - Validation split: 75 samples
  - Test split: 75 samples (held-out frozen split used for final model evaluation)
- **Evaluated Models:**
  - **Random Forest Baseline:** `RandomForestRegressor(n_estimators=300, random_state=42)` trained on 350 samples.
  - **Single-Timestep LSTM:** Sequential architecture `LSTM(64) -> Dense(32, ReLU) -> Dense(1, Linear)` trained for 100 epochs with Adam optimizer and batch size 32. Input shape `(batch_size, 1, 38)`.

---

## 2. Model Results

### Test Split Evaluation (Held-Out Test Set)

All comparable metrics are evaluated on the exact same frozen test split (`data/processed/X_test.csv` & `y_test.csv`).

| Model Architecture | Test MAE (kg/ha) | Test RMSE (kg/ha) | Test R² | Data Split / Notes |
|---|---:|---:|---:|---|
| **Mean Baseline** | N/A | N/A | N/A | Not provided in team handoff |
| **Linear Regression** | N/A | N/A | N/A | Not provided in team handoff |
| **Random Forest (Official)** | **1085.6482** | **1202.6050** | **0.0214** | Frozen test set (75 samples) |
| **Single-Timestep LSTM (Sreenidhi)** | 1714.7845 | 2073.7853 | -1.9100 | Frozen test set (1x38 input shape, no temporal claim) |
| **Older LSTM (Unfrozen Run)** | 4099.0000 | 4275.0000 | -11.3700 | *non-frozen data, not comparable* |
| **Federated LSTM (FedAvg)** | N/A | N/A | N/A | Multi-round FL experiment pending |

### Validation Split Evaluation (Separate Labeled Table)

| Model Architecture | Val MAE (kg/ha) | Val RMSE (kg/ha) | Val R² | Notes |
|---|---:|---:|---:|---|
| **Random Forest (Official)** | 1040.7291 | 1215.5774 | -0.1431 | Frozen validation set (75 samples) |
| **Single-Timestep LSTM (Sreenidhi)** | 1436.8066 | 1785.1052 | N/A | Frozen validation set (best epoch 100) |

---

## 3. Performance Results & Security Overhead

### AES-256 Micro-Benchmark Summary (30 Runs)
- **Serialized Payload Size:** 1,756 bytes
- **Encrypted Payload Size:** 1,784 bytes (+28 bytes / +1.59% overhead)
- **Average Encryption Latency:** `0.0042 ms`
- **Average Decryption Latency:** `0.0033 ms`
- **Average Security Overhead:** `0.0074 ms`

### Task B: Federated Round Breakdown Table
*Notice: Multi-round Flower FL experiment results were not present in the handoff. All round metrics are recorded as N/A.*

| Round # | Total Round Time (s) | Encryption (ms) | Decryption (ms) | Serialization (ms) | Deserialization (ms) | FedAvg Time (ms) | Encrypted Update Size (B) | Client Training Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Average** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** |

---

## 4. Energy Consumption

Validated joule-level energy telemetry was not available in the current Windows execution environment, so energy consumption is reported as unavailable rather than estimated as measured energy.

- **Measured Joules:** `N/A`
- **Estimated Energy:** `N/A`

---

## 5. Key Observations

1. **Baseline Model Outperforms LSTM:** The conventional Random Forest baseline significantly outperforms the single-timestep LSTM on held-out test data (Test MAE of **1085.65 kg/ha** vs **1714.78 kg/ha**; Test R² of **0.0214** vs **-1.9100**).
2. **Absence of Temporal Sequence Dependencies:** The input data is structured as 1 timestep x 38 features `(batch_size, 1, 38)`. Because sequence length is 1, the LSTM acts as a dense non-linear regressor and does not learn temporal sequence dependencies.
3. **Data Split Consistency:** Older reported LSTM results (Test MAE 4099, RMSE 4275, R² -11.37) were obtained on non-frozen data splits and are explicitly marked as non-comparable to the frozen split baseline evaluation.
4. **Negligible Cryptographic Overhead:** Micro-benchmarking confirms AES-256 encryption and decryption add an average security latency overhead of only **0.0074 ms** per 1.75 KB update payload.
5. **Minimal Bandwidth Payload Expansion:** Payload size increases by only 28 bytes (+1.59%) after AES-256 GCM/CBC encryption.
6. **Energy Telemetry Standard:** Due to host Windows OS telemetry limitations, energy consumption is strictly reported as `N/A` rather than fabricating synthetic energy measurements.
7. **Pending FL Handoff:** Multi-round Flower federated learning execution output files are pending from the upstream team.

---

## 6. Limitations

1. **Pending Multi-Round FL Output:** Round-by-round federated training metrics, client weight update logs, and FedAvg aggregation times could not be computed due to missing multi-round FL result files.
2. **Tabular Data Format:** The dataset structure (1 timestep x 38 features) limits the theoretical advantage of recurrent neural network (LSTM) architectures over decision tree ensembles.
3. **Hardware Energy Measurement:** Physical power meters and RAPL interfaces were unavailable in the Windows host execution environment.
4. **Baseline Handoff Scope:** Test split metrics for Mean and Linear Regression baselines were omitted in the initial handoff files.
