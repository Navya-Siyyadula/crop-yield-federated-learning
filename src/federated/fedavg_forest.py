"""
Federated aggregation for tree-ensemble clients (Random Forest).

Random Forest has no shared weight vector, so numeric FedAvg does not
apply. Instead we do "federated ensembling": each client contributes its
trained trees to a single global forest, weighted by how much local data
it trained on (n_samples) - a client with more data contributes more trees
to the merged forest.

This file provides:
  - get_client_update(model, n_samples)  -> the (update, n_samples) pair
    your FedAvg loop can consume, in place of (model_weights, n_samples)
  - aggregate_federated_forest(client_updates) -> one merged
    RandomForestRegressor built from all clients' trees
"""

import copy
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

EDGE_MODELS_DIR = Path("edge_models")
SHARED_DIR = Path("agri_zip")


def get_client_update(model: RandomForestRegressor, n_samples: int) -> dict:
    """The per-client 'update' to send to the server.

    Instead of a weight tensor, the update is the client's trained trees
    plus metadata needed to merge them safely.
    """
    return {
        "trees": model.estimators_,          # list of fitted DecisionTreeRegressor
        "n_features_in_": model.n_features_in_,
        "n_samples": n_samples,
    }


def aggregate_federated_forest(client_updates: list[dict],
                                total_trees: int = 400) -> RandomForestRegressor:
    """Merge client tree updates into one global forest.

    Trees are sampled from each client in proportion to n_samples, so
    clients with more local data contribute more trees - this is the
    tree-ensemble analogue of FedAvg's sample-weighted averaging.
    """
    n_features = client_updates[0]["n_features_in_"]
    total_samples = sum(u["n_samples"] for u in client_updates)

    merged_trees = []
    for u in client_updates:
        share = u["n_samples"] / total_samples
        n_take = max(1, round(share * total_trees))
        # take n_take trees from this client (cycle if it has fewer than n_take)
        client_trees = u["trees"]
        picked = [client_trees[i % len(client_trees)] for i in range(n_take)]
        merged_trees.extend(copy.deepcopy(picked))
        print(f"Client contributed {n_take} trees "
              f"(n_samples={u['n_samples']}, share={share:.2%})")

    # Build a global RandomForestRegressor shell and inject the merged trees
    global_model = RandomForestRegressor(n_estimators=len(merged_trees))
    global_model.estimators_ = merged_trees
    global_model.n_features_in_ = n_features
    global_model.n_outputs_ = 1
    global_model.estimator_ = client_updates[0]["trees"][0]  # template estimator, sklearn needs this attr

    return global_model


def main():
    # Load each client's trained model + local sample count
    n_samples_by_server = {1: 88, 2: 88, 3: 88, 4: 86}  # from your local_train_final.py run
    client_updates = []
    for server_id, n in n_samples_by_server.items():
        model = joblib.load(EDGE_MODELS_DIR / f"edge_server_{server_id}_rf_model.joblib")
        client_updates.append(get_client_update(model, n))

    global_model = aggregate_federated_forest(client_updates, total_trees=400)

    # Evaluate the merged global forest on the shared test set
    X_test = pd.read_csv(SHARED_DIR / "X_test.csv")
    y_test = pd.read_csv(SHARED_DIR / "y_test.csv").squeeze()
    preds = global_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"\nGlobal aggregated forest test MAE: {mae:.2f}")

    joblib.dump(global_model, "global_federated_rf_model.joblib")
    print("Saved global model to global_federated_rf_model.joblib")


if __name__ == "__main__":
    main()