# Latency Measurement & Performance Methodology

## 1. Overview & Objective
This document details the latency measurement framework, performance benchmarking methodology, and experimental findings for the Federated Learning Crop Yield Prediction system.

The objective is to quantify:
1. Model predictive performance on frozen test data splits.
2. Micro-benchmark overhead of AES-256 payload encryption and decryption.
3. Federated end-to-end round latency components (client training time, serialization time, encryption/decryption overhead, and FedAvg aggregation time).

---

## 2. Experimental Environment & System Hardware

- **Operating System:** Microsoft Windows 11 (Build 10.0.26200)
- **Host Processor:** Intel(R) Core(TM) Architecture (4 logical CPU cores)
- **Memory (RAM):** 16+ GB
- **Runtime & Software Stack:**
  - Python: `3.14.7` (compatible with 3.12+ project standard)
  - TensorFlow / Keras: `2.21.0` / Keras 3
  - Flower (FL Framework): `1.38.0` / `1.8+`
  - Cryptography Library: `50.0.1` (`46.0.7` compatible)
  - Data Processing: Pandas, NumPy, Scikit-learn, Matplotlib

---

## 3. Client & Model Configuration

- **Simulated Clients:** Edge farm nodes loaded with local partitioned agricultural dataset chunks.
- **Model Architecture (LSTM):**
  - Input Layer: Shape `(batch_size, 1, 38)` — 1 timestep x 38 preprocessed tabular features.
  - Core Layer: `LSTM(64 units)`
  - Hidden Layer: `Dense(32 units, activation='relu')`
  - Output Layer: `Dense(1 unit, activation='linear')` predicting target `yield_kg_per_hectare`.
- **Training Hyperparameters:**
  - Epochs: 100 local epochs
  - Batch Size: 32
  - Optimizer: Adam
  - Loss Function: Mean Squared Error (MSE)
  - Primary Metric: Mean Absolute Error (MAE)

---

## 4. Security & Encryption Mechanism (AES-256)

Model weights are serialized into binary updates prior to transmission. To guarantee update confidentiality in transit:
- **Cipher:** AES-256 symmetric encryption (256-bit key).
- **Mechanism:** Serialized numpy weight arrays are encrypted on the client side before submission and decrypted on the central server prior to FedAvg aggregation.
- **Key Safety Rule:** `FL_AES_KEY` is maintained exclusively in ephemeral environment variables and is strictly excluded from repository code, logs, and committed files.

---

## 5. Latency & Encryption Micro-Benchmark Findings

A micro-benchmark (`measure_overhead.py`) evaluated 30 independent encryption/decryption cycles on a standard 1,756-byte serialized update payload.

### Empirical Overhead Results:
- **Serialized Update Size:** 1,756 bytes
- **Encrypted Update Size:** 1,784 bytes (+28 bytes / +1.59% overhead)
- **Average Encryption Latency:** `0.0042 ms`
- **Average Decryption Latency:** `0.0033 ms`
- **Total Security Latency Overhead:** `0.0074 ms`

*Observation:* Cryptographic overhead adds under `0.01 ms` of processing latency per 1.75 KB update, proving AES-256 encryption is negligible relative to network transmission and local model training times.

---

## 6. Multi-Round FL Latency Status

Multi-round Flower FL experiment results (round-by-round time, client training latency breakdown, FedAvg aggregation duration) are not present in the current repository handoff. Per-round network and execution metrics are accordingly marked as `N/A`.
