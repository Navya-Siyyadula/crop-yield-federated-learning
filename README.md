# Crop Yield Federated Learning

This project studies crop-yield regression with a single-timestep LSTM, Flower federated learning, a logical edge/cloud comparison, AES-256-GCM protected model updates, and NSGA-II resource optimization.

## Dataset and method

The frozen processed dataset contains 350 training, 75 validation, and 75 test rows, with 38 features and target `yield_kg_per_hectare`. The model input is `(samples, 1, 38)` and the architecture is `LSTM(64) → Dense(32, ReLU) → Dense(1)`.

One `StandardScaler` is fit on all 350 training targets and shared by four Flower clients (88/88/88/86 samples). Validation and test targets do not fit the scaler; predictions are inverse-transformed before scoring in kg/ha. Flower FedAvg runs 20 rounds per candidate with seed 42. AES-256-GCM provides confidentiality and authenticated integrity for update payloads. NSGA-II minimizes measured latency and CodeCarbon-estimated energy as separate objectives; MAE, RMSE, and R² are evaluation metrics only.

## Quickstart

The repository already contains the final measured results, so no experiment run is needed to inspect them. The full comparison is computationally expensive (about 28 minutes on the recorded machine).

```bash
# 1. Clone and enter the repository
git clone https://github.com/Navya-Siyyadula/crop-yield-federated-learning.git
cd crop-yield-federated-learning

# 2. Create and activate an environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install the tested experiment dependencies
python -m pip install -r requirements-experiment.txt

# 4. Run tests
python -m pytest -q

# 5. Reproduce the complete comparison only if required (expensive)
python scripts/run_final_comparison.py --rounds 20 --central-epochs 20 --population 6 --generations 3
```

Inspect existing outputs under `results/`. The final narrative is in [docs/final_report.md](docs/final_report.md); protocol details are in [docs/experiments.md](docs/experiments.md).

## Final results

| Approach | MAE (kg/ha) | RMSE (kg/ha) | R² | Latency (s) | Energy (J, estimated) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centralized LSTM | 1,077.62 | 1,198.57 | 0.0280 | 4.8486 | 88.9362 |
| Edge–Cloud LSTM | 1,123.96 | 1,243.78 | -0.0468 | 12.5459 | 120.9701 |
| Federated LSTM | 1,094.43 | 1,224.33 | -0.0143 | 111.8122 | 1,039.7282 |
| EMO-selected Federated LSTM | 1,093.51 | 1,215.88 | -0.0003 | 110.1596 | 1,024.7889 |

NSGA-II evaluated 13 unique configurations. Its observed Pareto solution was `candidate_012` (generation 2, one local epoch, batch size 64), selected by normalized Euclidean distance to the ideal point. The engineering pipeline completed successfully, but predictive performance is modest and R² is near zero; these results do not support a high-accuracy claim.

## Limitations

The edge/cloud and federated runs are logical single-host loopback simulations, not physical deployment or WAN measurements. Latency is measured execution time. CodeCarbon supplies estimated energy, not electrical meter readings. The LSTM sees one timestep and does not learn multi-step temporal dependencies. Findings are limited to this frozen dataset and split. See [EMO methodology](docs/emo_methodology.md), [security](docs/security.md), and [performance methodology](docs/performance/latency_methodology.md).

## Project structure

`src/` contains the model, clients, federation, security, performance, and evaluation code; `scripts/` contains command-line entry points; `tests/` contains the test suite; `docs/` contains the methodology and final report; and `results/` contains experiment artifacts.

## License

See [LICENSE](LICENSE).
