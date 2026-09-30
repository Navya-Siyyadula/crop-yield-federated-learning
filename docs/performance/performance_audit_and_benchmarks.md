# Performance Audit and Experimental Results

## 1. Overview
This document summarizes the audit of performance data, baseline ML model comparisons, security overhead benchmarks, and federated learning execution status.

---

## 2. Generated Data Audit Summary

| File Path | Purpose | Format / Columns | Row Count | Missing Values | Granularity |
|---|---|---|---:|---|---|
| `results/baseline_metrics.json` | Official Random Forest model baseline metrics | JSON (train, val, test metrics) | 24 lines | None | Global baseline |
| `results/encryption_overhead.csv` | AES-256 micro-benchmark timings (1.75 KB update) | CSV (run, size, enc_ms, dec_ms, overhead_ms) | 30 data rows | None | Micro-benchmark |
| `results/Sreenidhi_LSTM_run_log.csv` | Centralized single-timestep LSTM training log | CSV (run_id, params, mse, mae, rmse, r2) | 1 data row | None | Centralized LSTM |
| `data/processed/*.csv` | Preprocessed tabular feature and target splits | CSV (X_train, X_val, X_test, y_train, y_val, y_test) | 350 / 75 / 75 | None | Global splits |

*Stale Data Flag:* `results/encryption_overhead.csv` contains valid standalone micro-benchmark metrics for a fixed 1756-byte payload, but predates any multi-round Flower federated experiment.

---

## 3. Model Benchmark Comparison (Test Split)

All models in this comparison table are evaluated on the exact same frozen test split (`data/processed/X_test.csv` & `y_test.csv`, 75 samples).

| Model Architecture | Test MAE (kg/ha) | Test RMSE (kg/ha) | Test R² | Status / Notes |
|---|---:|---:|---:|---|
| **Mean Baseline** | N/A | N/A | N/A | Not provided in team handoff |
| **Linear Regression** | N/A | N/A | N/A | Not provided in team handoff |
| **Random Forest (Official)** | **1085.6482** | **1202.6050** | **0.0214** | Evaluated on frozen test split |
| **Single-Timestep LSTM (Sreenidhi)** | 1714.7845 | 2073.7853 | -1.9100 | Evaluated on frozen test split (1x38 shape) |
| **Older LSTM (Unfrozen Run)** | 4099.0000 | 4275.0000 | -11.3700 | *Non-frozen data, not comparable* |
| **Federated LSTM (FedAvg)** | N/A | N/A | N/A | Multi-round FL experiment pending |

### Validation Split Metrics (Separate Table)
| Model Architecture | Val MAE (kg/ha) | Val RMSE (kg/ha) | Val R² | Notes |
|---|---:|---:|---:|---|
| **Random Forest (Official)** | 1040.7291 | 1215.5774 | -0.1431 | Frozen validation set (75 samples) |
| **Single-Timestep LSTM (Sreenidhi)** | 1436.8066 | 1785.1052 | N/A | Frozen validation set (best epoch 100) |

---

## 4. Key Performance Observations

1. **Baseline Model Superiority:** The official Random Forest baseline achieved a Test MAE of 1085.65 kg/ha and Test R² of 0.0214, outperforming the single-timestep LSTM (Test MAE 1714.78 kg/ha, Test R² -1.9100).
2. **Dataset Structure & Temporal Dependencies:** The dataset contains tabular agricultural indicators formatted as 1 timestep x 38 features `(batch_size, 1, 38)`. The LSTM model does not learn temporal sequence dependencies because sequence depth is 1.
3. **Encryption Efficiency:** AES-256 GCM/CBC encryption adds an average of 0.0074 ms security overhead per 1.75 KB update payload, with +28 bytes (+1.59%) size overhead.
4. **Energy Measurement Limitation:** Validated joule-level energy telemetry was not available in the current Windows execution environment, so energy consumption is reported as unavailable rather than estimated as measured energy.
5. **Pending FL Results:** Multi-round Flower federated experiment output files are missing from the handoff, so all multi-round round latency, client update sizes, and FL test metrics are recorded as `N/A`.
