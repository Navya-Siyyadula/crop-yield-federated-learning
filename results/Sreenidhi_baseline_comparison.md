# Sreenidhi LSTM vs Random Forest Baseline

## Experiment setup
- Task: Regression
- Target: `yield_kg_per_hectare`
- Processed features: 38
- Train split: 350 samples
- Validation split: 75 samples
- Test split: 75 samples
- Same frozen train/validation/test splits used for both models.

## Test-set comparison

| Metric | Random Forest baseline | Sreenidhi LSTM |
|---|---:|---:|
| MAE (kg/ha) | 1085.6482 | 1714.7845 |
| RMSE (kg/ha) | 1202.6050 | 2073.7853 |
| R² | 0.0214 | -1.9100 |

## Interpretation
The official Random Forest baseline has lower test MAE and RMSE than the Sreenidhi LSTM run. Its test R² is also higher. Therefore, in this recorded experiment, the LSTM did not outperform the conventional Random Forest baseline on the held-out test set.

The LSTM result is reported without changing the test set or tuning against the test results. The LSTM was selected using validation MAE and evaluated on the held-out test set once after model selection.

## LSTM run
- Architecture: LSTM(64) -> Dense(32, ReLU) -> Dense(1)
- Optimizer: Adam
- Loss: MSE
- Metric: MAE
- Input shape: `(batch_size, 1, 38)`
- Best validation epoch: 100
- Test MAE: 1714.7845 kg/ha
- Test RMSE: 2073.7853 kg/ha
- Test R²: -1.9100

## Official baseline run
- Model: `RandomForestRegressor`
- Estimators: 300
- Random state: 42
- Test MAE: 1085.6482 kg/ha
- Test RMSE: 1202.6050 kg/ha
- Test R²: 0.0214

These are the measured results from the supplied runs; no accuracy value has been invented or forced.
