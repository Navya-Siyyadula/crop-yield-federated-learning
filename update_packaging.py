"""
Package trained Random Forest model updates for the Encryption team.

Expected packaged update format:

{
    "trees": [...],
    "n_features_in_": 38,
    "n_samples": ...
}

The trained models are produced by:
train_separate_random_forests.py
"""

from pathlib import Path
import json
import joblib


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# update_packaging.py is located at:
# project_root/src/edge/update_packaging.py
#
# parents[0] = edge
# parents[1] = src
# parents[2] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLIENT_IDS = [1, 2, 3, 4]


# ---------------------------------------------------------------------
# Tree conversion
# ---------------------------------------------------------------------

def convert_tree(tree):
    """
    Convert one sklearn DecisionTreeRegressor into a
    serialization-ready Python dictionary.

    NumPy arrays are converted into normal Python lists.
    """

    tree_state = tree.tree_.__getstate__()

    return {
        "max_depth": int(tree_state["max_depth"]),
        "node_count": int(tree_state["node_count"]),
        "nodes": tree_state["nodes"].tolist(),
        "values": tree_state["values"].tolist(),
    }


# ---------------------------------------------------------------------
# Package one client
# ---------------------------------------------------------------------

def package_client_update(client_id):
    """
    Load and package the trained Random Forest for one client.

    Parameters
    ----------
    client_id : int
        Client number: 1, 2, 3, or 4.

    Returns
    -------
    dict
        Packaged model update containing:
            - trees
            - n_features_in_
            - n_samples
    """

    client_dir = PROJECT_ROOT / f"client_{client_id}"

    model_path = (
        client_dir /
        f"client_{client_id}_random_forest.joblib"
    )

    metrics_path = (
        client_dir /
        f"client_{client_id}_metrics.json"
    )

    # Check that the trained model exists.
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found for client {client_id}: "
            f"{model_path}"
        )

    # Check that the metrics file exists.
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found for client {client_id}: "
            f"{metrics_path}"
        )

    # Load trained Random Forest.
    model = joblib.load(model_path)

    # Load training metadata.
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    # Make sure the loaded model has Random Forest estimators.
    if not hasattr(model, "estimators_"):
        raise TypeError(
            f"Client {client_id} model does not contain "
            "Random Forest estimators."
        )

    # Make sure the model contains the number of input features.
    if not hasattr(model, "n_features_in_"):
        raise AttributeError(
            f"Client {client_id} model does not contain "
            "n_features_in_."
        )

    # Convert all decision trees.
    trees = [
        convert_tree(tree)
        for tree in model.estimators_
    ]

    # The previous training script records the number of
    # training samples as "n_train" in the metrics JSON.
    if "n_train" not in metrics:
        raise KeyError(
            f"Client {client_id} metrics file does not contain "
            "'n_train'."
        )

    n_samples = int(metrics["n_train"])

    # Create the agreed package.
    packaged_update = {
        "trees": trees,
        "n_features_in_": int(model.n_features_in_),
        "n_samples": n_samples,
    }

    return packaged_update


# ---------------------------------------------------------------------
# Package all four clients
# ---------------------------------------------------------------------

def package_all_client_updates():
    """
    Package model updates for all four clients.

    Returns
    -------
    dict
        Dictionary containing one packaged update per client.
    """

    updates = {}

    for client_id in CLIENT_IDS:
        updates[client_id] = package_client_update(client_id)

    return updates


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_update(update, client_id):
    """
    Check that a packaged client update has the expected
    structure and contains non-empty data.
    """

    required_keys = {
        "trees",
        "n_features_in_",
        "n_samples",
    }

    # Check top-level type.
    if not isinstance(update, dict):
        raise TypeError(
            f"Client {client_id}: update must be a dictionary."
        )

    # Check required fields.
    missing_keys = required_keys - set(update.keys())

    if missing_keys:
        raise ValueError(
            f"Client {client_id}: missing keys: "
            f"{sorted(missing_keys)}"
        )

    # Check trees.
    if not isinstance(update["trees"], list):
        raise TypeError(
            f"Client {client_id}: 'trees' must be a list."
        )

    if len(update["trees"]) == 0:
        raise ValueError(
            f"Client {client_id}: 'trees' cannot be empty."
        )

    # Check feature count.
    if update["n_features_in_"] <= 0:
        raise ValueError(
            f"Client {client_id}: 'n_features_in_' "
            "must be greater than zero."
        )

    # Check sample count.
    if update["n_samples"] <= 0:
        raise ValueError(
            f"Client {client_id}: 'n_samples' "
            "must be greater than zero."
        )

    # Check every tree.
    for tree_index, tree in enumerate(update["trees"]):

        if not isinstance(tree, dict):
            raise TypeError(
                f"Client {client_id}, tree {tree_index}: "
                "tree must be a dictionary."
            )

        required_tree_keys = {
            "max_depth",
            "node_count",
            "nodes",
            "values",
        }

        missing_tree_keys = (
            required_tree_keys - set(tree.keys())
        )

        if missing_tree_keys:
            raise ValueError(
                f"Client {client_id}, tree {tree_index}: "
                f"missing keys: {sorted(missing_tree_keys)}"
            )

        if not tree["nodes"]:
            raise ValueError(
                f"Client {client_id}, tree {tree_index}: "
                "nodes cannot be empty."
            )

        if not tree["values"]:
            raise ValueError(
                f"Client {client_id}, tree {tree_index}: "
                "values cannot be empty."
            )

    return True


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Package and validate all four client updates.
    """

    updates = package_all_client_updates()

    print("Model update packaging results")
    print("=" * 40)

    for client_id in CLIENT_IDS:

        update = updates[client_id]

        validate_update(update, client_id)

        print(
            f"Client {client_id}: "
            f"{len(update['trees'])} trees, "
            f"{update['n_features_in_']} features, "
            f"{update['n_samples']} training samples"
        )

    print("=" * 40)
    print("All 4 client updates packaged successfully.")

    return updates


if __name__ == "__main__":
    main()