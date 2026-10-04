"""Flower client for local training with encrypted LSTM model updates."""

from __future__ import annotations

from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd
from flwr.client import Client
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
)

from src.performance.latency import LatencyCollector
from src.security.key_manager import load_key_from_environment
from src.security.lstm_encrypted_update import (
    encrypted_weights_to_parameters,
    parameters_to_decrypted_weights,
)
from src.clients.local_training import train_local_model

FEATURE_COUNT = 38
CLIENT_ID_PATTERN = re.compile(r"^client_[0-9]{2,}$")


def load_client_dataset(client_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one client's existing processed CSV pair; never partitions or creates data."""
    client_dir = Path(client_dir)
    # Prefer the canonical train filenames when present. The frozen partitions
    # currently tracked by the repository use client_N_X/client_N_y names.
    match = re.fullmatch(r"client_([0-9]+)", client_dir.name)
    suffix = match.group(1) if match else None
    x_path = client_dir / "X_train.csv"
    y_path = client_dir / "y_train.csv"
    if suffix is not None:
        partition_x = client_dir / f"client_{int(suffix)}_X.csv"
        partition_y = client_dir / f"client_{int(suffix)}_y.csv"
        if x_path.is_file() and y_path.is_file():
            pass
        elif partition_x.is_file() and partition_y.is_file():
            x_path, y_path = partition_x, partition_y
    missing = [path for path in (x_path, y_path) if not path.is_file()]
    if missing:
        expected = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Client training data is missing: {expected}")

    features = pd.read_csv(x_path).to_numpy(dtype=np.float32)
    targets_frame = pd.read_csv(y_path)
    if targets_frame.shape[1] != 1:
        raise ValueError(f"Expected one target column in {y_path}")
    targets = targets_frame.iloc[:, 0].to_numpy(dtype=np.float32)

    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"Expected X_train with {FEATURE_COUNT} columns; got shape {features.shape}"
        )
    if len(features) == 0 or len(targets) == 0:
        raise ValueError("Client training dataset must contain at least one sample")
    if len(features) != len(targets):
        raise ValueError(
            f"Feature/target row counts differ: {len(features)} != {len(targets)}"
        )
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Client training data contains non-finite values")

    # Match src/ml/train_lstm.py: one timestep containing 38 processed features.
    return features.reshape(-1, 1, FEATURE_COUNT), targets


class EncryptedLSTMFlowerClient(Client):
    """Low-level Flower client preserving the bridge's opaque encrypted bytes."""

    def __init__(
        self,
        client_id: str,
        data_root: str | Path = "data/client_data",
        *,
        epochs: int = 1,
        batch_size: int = 32,
        model=None,
        key: bytes | None = None,
        target_mean: float = 0.0,
        target_scale: float = 1.0,
    ) -> None:
        if not CLIENT_ID_PATTERN.fullmatch(client_id):
            raise ValueError("client_id must look like client_01")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        self.client_id = client_id
        self.data_dir = Path(data_root) / client_id
        self.x_train, self.y_train = load_client_dataset(self.data_dir)
        self.target_mean = float(target_mean)
        self.target_scale = float(target_scale)
        if not np.isfinite(self.target_mean) or not np.isfinite(self.target_scale):
            raise ValueError("Shared target scaler parameters must be finite")
        if self.target_scale <= 0:
            raise ValueError("Shared target scaler scale must be positive")
        self.y_train_kg_per_hectare = self.y_train.copy()
        self.y_train = ((self.y_train - self.target_mean) / self.target_scale).astype(np.float32)
        self.epochs = epochs
        self.batch_size = batch_size
        self.key = key if key is not None else load_key_from_environment()
        if model is None:
            # Build the same architecture used by the existing encrypted server.
            from src.ml.lstm_model import build_lstm_model

            model = build_lstm_model(input_features=FEATURE_COUNT)
        self.model = model

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        _ = ins
        return GetPropertiesRes(
            status=Status(code=Code.OK, message=""),
            properties={"client_id": self.client_id},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = ins
        parameters = encrypted_weights_to_parameters(self.model.get_weights(), self.key)
        return GetParametersRes(
            status=Status(code=Code.OK, message=""),
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        # Flower Parameters contain the bridge's single opaque AES-GCM payload,
        # not a NumPy tensor list.
        decryption_started = perf_counter()
        global_weights = parameters_to_decrypted_weights(ins.parameters, self.key)
        global_decryption_ms = (perf_counter() - decryption_started) * 1000.0
        self.model.set_weights(global_weights)

        # A round/client-specific seed makes local shuffle initialization
        # reproducible in the local four-client experiment.
        import tensorflow as tf
        server_round = int(ins.config.get("server_round", 0))
        client_number = int(self.client_id.rsplit("_", 1)[1])
        tf.keras.utils.set_random_seed(42 + 1009 * server_round + client_number)

        latency = LatencyCollector()
        with latency.measure("local_training_ms"):
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                shuffle=False,
            )

        encrypted_update = encrypted_weights_to_parameters(
            self.model.get_weights(),
            self.key,
            latency=latency,
        )
        payload_size = sum(len(tensor) for tensor in encrypted_update.tensors)
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=encrypted_update,
            num_examples=len(self.y_train),
            metrics={
                "client_id": self.client_id,
                "training_latency_ms": latency.get("local_training_ms") or 0.0,
                "serialization_latency_ms": latency.get("serialization_ms") or 0.0,
                "encryption_latency_ms": latency.get("encryption_ms") or 0.0,
                "decryption_latency_ms": global_decryption_ms,
                "encrypted_update_size_bytes": payload_size,
                "target_scaler_mean": self.target_mean,
                "target_scaler_scale": self.target_scale,
            },
        )
class EdgeClient:
    """Represents one simulated edge client/farm."""

    def __init__(self, client_id, x_path, y_path):
        self.client_id = client_id
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)

    def load_data(self):
        X = pd.read_csv(self.x_path).values
        y = pd.read_csv(self.y_path).values
        return X, y

    def fit(self, initial_weights=None, epochs=1, batch_size=16):
        X, y = self.load_data()
        weights, num_samples = train_local_model(
            X=X,
            y=y,
            initial_weights=initial_weights,
            epochs=epochs,
            batch_size=batch_size,
        )
        return weights, num_samples
