# Project status

## Project status

COMPLETE — experimental implementation and documentation frozen.

## Dataset

Frozen processed split: 350 train, 75 validation, and 75 test samples; 38 features; target `yield_kg_per_hectare`.

## Model

Single-timestep LSTM regression: input `(1, 38)`, LSTM(64), Dense(32, ReLU), Dense(1). One target `StandardScaler` is fit only on the 350 training targets and shared by all clients; predictions are inverse-transformed before scoring.

## Federated learning

Flower FedAvg, four clients (88/88/88/86 samples), 20 rounds per candidate, seed 42, executed as a logical single-host loopback simulation.

## Security

AES-256-GCM model-update protection, providing confidentiality and authenticated integrity. RSA key exchange and homomorphic encryption are not implemented in this path.

## Optimization

NSGA-II minimizes measured latency and CodeCarbon-estimated energy as two separate objectives. Prediction metrics are evaluation measures, not optimization objectives.

## Final Pareto candidate

`candidate_012`, generation 2, one local epoch, batch size 64; latency 110.1596 s and estimated energy 1,024.7889 J.

## Final predictive metrics

| Approach | MAE (kg/ha) | RMSE (kg/ha) | R² |
| --- | ---: | ---: | ---: |
| Centralized LSTM | 1,077.62 | 1,198.57 | 0.0280 |
| Edge–Cloud LSTM | 1,123.96 | 1,243.78 | -0.0468 |
| Federated LSTM | 1,094.43 | 1,224.33 | -0.0143 |
| EMO-selected Federated LSTM | 1,093.51 | 1,215.88 | -0.0003 |

## Final performance

| Approach | Latency (s) | Energy (J, estimated) |
| --- | ---: | ---: |
| Centralized LSTM | 4.8486 | 88.9362 |
| Edge–Cloud LSTM | 12.5459 | 120.9701 |
| Federated LSTM | 111.8122 | 1,039.7282 |
| EMO-selected Federated LSTM | 110.1596 | 1,024.7889 |

## Tests

33 passed, 0 failed, 0 skipped (two protobuf deprecation warnings).

## Known limitations

- Single-timestep LSTM; it does not model multi-step temporal dependencies.
- Predictive performance is modest, with R² near zero.
- Edge/cloud and FL execution are logical single-host simulations.
- CodeCarbon values are estimated energy, not electrical meter measurements.
- Conclusions are limited by the frozen dataset and split.

## Future work

Potentially investigate models better suited to tabular data and richer feature engineering. This is future work and was not performed as part of the frozen project.
