# Final federated LSTM and EMO experiment

## Protocol and verification

The final run used the frozen 350/75/75 split, 38 features, target `yield_kg_per_hectare`, seed 42, Python 3.12.14, TensorFlow 2.18.0, Flower 1.8.0, and CodeCarbon 3.3.1 on Windows 10/Intel CPU. One `StandardScaler` was fit on all 350 training targets only (mean 4,069.6537; scale 1,165.5102 kg/ha) and the same parameters were used by all four clients. Clients trained on normalized targets. Validation/global and final predictions were inverse-transformed before scoring in kg/ha. Test files were opened only after NSGA-II selected the final candidate.

The centralized comparison used the same scaler and restored the best validation-loss checkpoint (epoch 1 of 20). Federated candidates used fixed local epochs per round without local validation checkpointing. One logical edge/cloud Flower round used 20 local epochs and batch size 32. The 4-client Flower server/client processes ran over loopback on a single host.

## Final test comparison

| Approach | MAE (kg/ha) | RMSE (kg/ha) | R² | Latency (s) | Estimated energy (J) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centralized LSTM (best validation checkpoint) | 1,077.62 | 1,198.57 | 0.0280 | 4.8486 | 88.9362 |
| Edge–Cloud LSTM (one-round logical simulation) | 1,123.96 | 1,243.78 | -0.0468 | 12.5459 | 120.9701 |
| Federated LSTM baseline (`candidate_001`, 1 epoch, batch 32) | 1,094.43 | 1,224.33 | -0.0143 | 111.8122 | 1,039.7282 |
| EMO-selected Federated LSTM (`candidate_012`, 1 epoch, batch 64) | 1,093.51 | 1,215.88 | -0.0003 | 110.1596 | 1,024.7889 |

All federated test predictions were inverse-transformed with the shared training scaler. Selected-model metrics were independently recomputed with sklearn from the same original-unit predictions. The results are modest: the centralized R² is about 0.028 and the EMO-selected federated R² is near zero. They do not support a claim of high predictive accuracy. The existing Random Forest baseline scored test R² 0.0214 on this split, slightly below the centralized LSTM and slightly above the selected federated result.

## Federated execution and EMO

The runner evaluated **13 unique configurations** across population 6 and 3 generations. Repeated genotypes were skipped, so not all 15 possible `(local_epochs, batch_size)` combinations were measured. Each measured candidate completed 20 Flower FedAvg rounds with four clients and 350 samples per round: 260 candidate-round rows and 1,040 client records. Every round received four client updates with zero failures. AES-256-GCM protected serialized updates. Per-client telemetry shows the same scaler mean/scale for all clients.

Local epochs ranged from 1–5 and batch sizes from 16/32/64. Candidate latency is the sum of actual measured round durations. Candidate energy is the sum of CodeCarbon whole-machine software estimates; all 260 candidate-round records have estimated energy. The final candidate search took about 27 minutes 38 seconds wall time. This is a single-host simulation, not physical edge/WAN measurement, and energy is estimated rather than wall-power telemetry.

The observed Pareto front contains one point: **candidate_012**, generation 2, 1 local epoch, batch size 64, latency **110.1596 s**, estimated energy **1,024.7889 J**. It is the observed ideal point for both objectives, so its normalized latency and energy are both 0.0. Selection used normalized Euclidean distance to the ideal point. Prediction metrics and test data did not affect candidate ranking or selection. Candidate 001 is the fixed federated baseline; candidate 012 is a distinct trained configuration.

The one-round preflight verified the shared scaler fit count (350), identical client scaler parameters, model input/output shapes, inverse-transform path, AES-256-GCM encryption/decryption, four client updates, measured latency, and CodeCarbon energy estimate before the full search.

## Security and limitations

AES-256-GCM provides confidentiality and authenticated integrity for serialized updates. The implementation does not use RSA and does not provide a production multi-party key-management protocol. All clients/server ran locally over loopback. The LSTM consumes a single timestep `(samples, 1, 38)`; it does not learn multi-step temporal dependencies.

## Reproduction and artifacts

Run from the repository root:

```powershell
python -m pip install -r requirements-experiment.txt
python -m pytest -q
python scripts/run_final_comparison.py --rounds 20 --central-epochs 20 --population 6 --generations 3
```

The completed run used `.venv\Scripts\python.exe`. The pre-run versions of overwritten final artifacts are preserved in `results/backups/pre_shared_target_scaler_20261004/` with SHA-256 verification in `manifest.json`.

Final outputs include `final_comparison.csv`, `emo_pareto_solutions.csv`, `fl_round_metrics.csv`, `client_performance.csv`, `final_experiment_config.json`, `final_experiment.log`, `latency_vs_energy_pareto.png`, `model_vs_r2.png`, and other comparison/round plots. Centralized diagnostic files `lstm_target_scaled_results.json` and `lstm_target_scaled_best_checkpoint_results.json` were preserved.

The full test suite passed: **33 passed, 0 failed, 0 skipped**, with two protobuf deprecation warnings.
