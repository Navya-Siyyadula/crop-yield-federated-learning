# Crop Yield Federated Learning

A federated learning system for predicting crop yield across multiple, non-colocated data sources (clients), without centralizing raw agricultural data. Includes an encrypted FL variant, performance/latency benchmarking, and a monitoring dashboard.

## Overview

Traditional ML pipelines require pooling all data centrally, which isn't always feasible or desirable for distributed agricultural data (privacy, bandwidth, ownership). This project trains a shared crop-yield model across multiple simulated clients using **Federated Averaging (FedAvg)**, comparing it against a centralized baseline, and optionally applying **homomorphic encryption** to protect model updates in transit.

## Project Structure

```
config/        # YAML configs for model, clients, FL rounds, security
data/          # raw -> interim -> cleaned -> processed, plus per-client splits
notebooks/     # exploratory analysis, from EDA through FL experiments
src/           # core library code
  data/            data loading, cleaning, validation, splitting
  preprocessing/   encoding, scaling, feature transforms
  models/          baseline + trainable model, train/predict/evaluate
  clients/         client simulation, local training
  federated/       FL server, FedAvg, round orchestration
  security/        encryption/decryption, key management
  performance/     timing, latency, energy metrics
  evaluation/      metrics, comparisons, reporting
experiments/   # one folder per experiment axis (centralized, edge, federated,
               # encryption, iid vs non-iid, client count, round count)
results/       # metrics, latency, energy, trained models, figures, tables
dashboard/     # Streamlit app for visualizing training/results
scripts/       # CLI entry points to run each stage end-to-end
tests/         # unit tests, mirrors src/ layout
docs/          # architecture, dataset, ML, FL, security, performance write-ups
```

## Getting Started

```bash
# 1. Clone and enter the repo
git clone https://github.com/Navya-Siyyadula/crop-yield-federated-learning.git
cd crop-yield-federated-learning

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template
cp .env.example .env
```

## Typical Workflow

```bash
python scripts/prepare_data.py        # clean + preprocess raw data
python scripts/create_clients.py      # partition data across clients
python scripts/train_centralized.py   # baseline model
python scripts/run_federated.py       # federated training (FedAvg)
python scripts/run_encrypted_fl.py    # federated training with encryption
python scripts/generate_results.py    # aggregate metrics into results/
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

Run tests:

```bash
pytest tests/ -v
```

## Team

See `docs/team/responsibility_matrix.md` for role breakdown and `docs/team/execution_book.md` for the project plan.

## License

See `LICENSE`.
