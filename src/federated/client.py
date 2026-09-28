"""
Flower client for encrypted LSTM Federated Learning.

Uses the low-level Flower Client API so encrypted model-update bytes
are never automatically deserialized as NumPy arrays.
"""

import argparse
from pathlib import Path

import flwr as fl
import numpy as np
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Status,
)

from src.ml.lstm_model import build_lstm_model
from src.security.key_manager import load_key_from_environment
from src.security.lstm_encrypted_update import (
    encrypted_weights_to_parameters,
    parameters_to_decrypted_weights,
)


CLIENT_DATA_DIR = Path("data/client_data")
VALIDATION_DIR = Path("data/processed")

INPUT_FEATURES = 38
LOCAL_EPOCHS = 1
BATCH_SIZE = 32


class EncryptedLSTMClient(fl.client.Client):

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.key = load_key_from_environment()

        client_dir = CLIENT_DATA_DIR / f"client_{client_id:02d}"

        self.X = np.loadtxt(
            client_dir / f"client_{client_id}_X.csv",
            delimiter=",",
            skiprows=1,
        )

        self.y = np.loadtxt(
            client_dir / f"client_{client_id}_y.csv",
            delimiter=",",
            skiprows=1,
        )

        self.X = np.asarray(self.X, dtype=np.float32).reshape(
            -1, 1, INPUT_FEATURES
        )
        self.y = np.asarray(self.y, dtype=np.float32).reshape(-1)

        self.model = build_lstm_model(INPUT_FEATURES)

        self.X_val = None
        self.y_val = None

        val_x_path = VALIDATION_DIR / "X_val.csv"
        val_y_path = VALIDATION_DIR / "y_val.csv"

        if val_x_path.exists() and val_y_path.exists():
            self.X_val = np.loadtxt(
                val_x_path,
                delimiter=",",
                skiprows=1,
            ).astype(np.float32).reshape(-1, 1, INPUT_FEATURES)

            self.y_val = np.loadtxt(
                val_y_path,
                delimiter=",",
                skiprows=1,
            ).astype(np.float32).reshape(-1)

        print(
            f"Client {self.client_id}: "
            f"training data {self.X.shape}, {self.y.shape}"
        )

    def get_parameters(
        self,
        ins: GetParametersIns,
    ) -> GetParametersRes:

        encrypted_parameters = encrypted_weights_to_parameters(
            self.model.get_weights(),
            self.key,
        )

        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=encrypted_parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:

        print(
            f"Client {self.client_id}: "
            "starting local LSTM training..."
        )

        # IMPORTANT:
        # Do NOT call parameters_to_ndarrays().
        # The parameters contain encrypted raw bytes.
        weights = parameters_to_decrypted_weights(
            ins.parameters,
            self.key,
        )

        self.model.set_weights(weights)

        history = self.model.fit(
            self.X,
            self.y,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            shuffle=True,
        )

        updated_weights = self.model.get_weights()

        encrypted_parameters = encrypted_weights_to_parameters(
            updated_weights,
            self.key,
        )

        final_loss = float(history.history["loss"][-1])
        final_mae = float(history.history["mae"][-1])

        print(
            f"Client {self.client_id}: "
            f"training complete | loss={final_loss:.4f} "
            f"| MAE={final_mae:.4f}"
        )

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=encrypted_parameters,
            num_examples=len(self.X),
            metrics={
                "loss": final_loss,
                "mae": final_mae,
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:

        weights = parameters_to_decrypted_weights(
            ins.parameters,
            self.key,
        )

        self.model.set_weights(weights)

        if self.X_val is None:
            return EvaluateRes(
                status=Status(code=Code.OK, message="No validation data"),
                loss=0.0,
                num_examples=0,
                metrics={},
            )

        loss, mae = self.model.evaluate(
            self.X_val,
            self.y_val,
            verbose=0,
        )

        print(
            f"Client {self.client_id}: "
            f"validation loss={float(loss):.4f} "
            f"| MAE={float(mae):.4f}"
        )

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss),
            num_examples=len(self.X_val),
            metrics={
                "mae": float(mae),
            },
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server-id",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
    )

    parser.add_argument(
        "--server-address",
        default="127.0.0.1:8080",
    )

    args = parser.parse_args()

    client = EncryptedLSTMClient(args.server_id)

    fl.client.start_client(
        server_address=args.server_address,
        client=client,
    )
