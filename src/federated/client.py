
"""
Flower client for one simulated edge server.

Each client:
1. Loads its local training data.
2. Trains a Random Forest.
3. Serialises and AES-256 encrypts the trained model.
4. Sends the encrypted model through Flower.
5. Decrypts the global model received from the server.
"""

import argparse
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import flwr as fl

from src.security.encrypted_update import (
    encrypted_model_to_parameters,
    parameters_to_encrypted_model,
)

from src.security.key_manager import (
    load_key_from_environment,
)

from src.federated.rf_serialization import (
    empty_parameters,
)


EDGE_DIR = Path("edge_zip")
SHARED_DIR = Path("agri_zip")
RANDOM_STATE = 42


class RFEdgeClient(fl.client.NumPyClient):

    def __init__(self, server_id: int):

        self.server_id = server_id

        # Load the shared AES-256 key
        self.encryption_key = load_key_from_environment()

        # Load local training data
        self.X_train = pd.read_csv(
            EDGE_DIR / f"client_{server_id}_X.csv"
        )

        self.y_train = pd.read_csv(
            EDGE_DIR / f"client_{server_id}_y.csv"
        ).squeeze()

        # Load shared validation data
        self.X_val = pd.read_csv(
            SHARED_DIR / "X_val.csv"
        )

        self.y_val = pd.read_csv(
            SHARED_DIR / "y_val.csv"
        ).squeeze()

        self.model = None

    def get_parameters(self, config):

        if self.model is None:
            return empty_parameters().tensors

        return encrypted_model_to_parameters(
            self.model,
            self.encryption_key,
        ).tensors

    def fit(self, parameters, config):

        print(
            f"Client {self.server_id}: "
            "training Random Forest..."
        )

        # Train local Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
        )

        self.model.fit(
            self.X_train,
            self.y_train,
        )

        # Serialize + AES-256 encrypt the model
        params = encrypted_model_to_parameters(
            self.model,
            self.encryption_key,
        )

        num_examples = len(
            self.X_train
        )

        print(
            f"Client {self.server_id}: "
            f"trained on {num_examples} samples."
        )

        print(
            f"Client {self.server_id}: "
            "model encrypted successfully."
        )

        return (
            params.tensors,
            num_examples,
            {
                "server_id": self.server_id,
            },
        )

    def evaluate(self, parameters, config):

        # Convert Flower Parameters back into
        # the encrypted Random Forest model
        from flwr.common import Parameters

        encrypted_parameters = Parameters(
            tensor_type="",
            tensors=parameters,
        )

        global_model = parameters_to_encrypted_model(
            encrypted_parameters,
            self.encryption_key,
        )

        # Evaluate the decrypted global model
        predictions = global_model.predict(
            self.X_val
        )

        mae = mean_absolute_error(
            self.y_val,
            predictions,
        )

        print(
            f"Client {self.server_id}: "
            f"validation MAE = {mae:.2f}"
        )

        return (
            float(mae),
            len(self.X_val),
            {
                "server_id": self.server_id,
                "mae": float(mae),
            },
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server-id",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
    )

    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
    )

    args = parser.parse_args()

    client = RFEdgeClient(
        args.server_id
    ).to_client()

    fl.client.start_client(
        server_address=args.server_address,
        client=client,
    )
