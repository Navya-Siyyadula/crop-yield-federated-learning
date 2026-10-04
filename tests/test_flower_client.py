import numpy as np
import pandas as pd
import pytest
from flwr.common import FitIns

from src.clients.client import EncryptedLSTMFlowerClient, load_client_dataset
from src.security.encryption import generate_key
from src.security.lstm_encrypted_update import (
    encrypted_weights_to_parameters,
    parameters_to_decrypted_weights,
)


def write_client_data(client_dir, *, rows=3, features=38):
    client_dir.mkdir(parents=True)
    pd.DataFrame(np.ones((rows, features), dtype=np.float32)).to_csv(
        client_dir / "X_train.csv", index=False
    )
    pd.DataFrame({"yield_kg_per_hectare": np.arange(rows, dtype=np.float32)}).to_csv(
        client_dir / "y_train.csv", index=False
    )


def test_load_client_csv_pair_matches_lstm_input_shape(tmp_path):
    client_dir = tmp_path / "client_01"
    write_client_data(client_dir)

    features, targets = load_client_dataset(client_dir)

    assert features.shape == (3, 1, 38)
    assert features.dtype == np.float32
    assert targets.shape == (3,)


def test_load_frozen_partition_csv_names(tmp_path):
    client_dir = tmp_path / "client_02"
    client_dir.mkdir()
    pd.DataFrame(np.ones((2, 38), dtype=np.float32)).to_csv(
        client_dir / "client_2_X.csv", index=False
    )
    pd.DataFrame({"yield_kg_per_hectare": [1.0, 2.0]}).to_csv(
        client_dir / "client_2_y.csv", index=False
    )

    features, targets = load_client_dataset(client_dir)

    assert features.shape == (2, 1, 38)
    np.testing.assert_array_equal(targets, np.array([1.0, 2.0], dtype=np.float32))


def test_client_data_missing_files_or_invalid_feature_count_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="X_train.csv"):
        load_client_dataset(tmp_path / "empty_client")

    bad_dir = tmp_path / "bad_client"
    write_client_data(bad_dir, features=37)
    with pytest.raises(ValueError, match="38 columns"):
        load_client_dataset(bad_dir)


class FakeModel:
    def __init__(self):
        self.weights = [np.array([0.0], dtype=np.float32)]
        self.fit_args = None

    def get_weights(self):
        return [weight.copy() for weight in self.weights]

    def set_weights(self, weights):
        self.weights = [np.asarray(weight).copy() for weight in weights]

    def fit(self, features, targets, **kwargs):
        self.fit_args = (features.shape, targets.shape, kwargs)
        self.weights[0] += 1.0


def test_fit_decrypts_trains_encrypts_and_reports_fedavg_sample_count(tmp_path):
    client_dir = tmp_path / "client_01"
    write_client_data(client_dir)
    key = generate_key()
    model = FakeModel()
    client = EncryptedLSTMFlowerClient(
        "client_01",
        data_root=tmp_path,
        model=model,
        key=key,
        epochs=2,
        batch_size=4,
        target_mean=1.0,
        target_scale=2.0,
    )
    received_parameters = encrypted_weights_to_parameters(
        [np.array([4.0], dtype=np.float32)], key
    )

    result = client.fit(FitIns(parameters=received_parameters, config={}))

    assert result.num_examples == 3
    assert model.weights[0][0] == 5.0
    assert model.fit_args[0] == (3, 1, 38)
    assert model.fit_args[1] == (3,)
    np.testing.assert_allclose(client.y_train, [-0.5, 0.0, 0.5])
    assert model.fit_args[2]["epochs"] == 2
    assert result.metrics["client_id"] == "client_01"
    assert result.metrics["training_latency_ms"] >= 0
    assert result.metrics["encrypted_update_size_bytes"] == len(
        result.parameters.tensors[0]
    )
    assert result.metrics["target_scaler_mean"] == 1.0
    assert result.metrics["target_scaler_scale"] == 2.0
    recovered = parameters_to_decrypted_weights(result.parameters, key)
    np.testing.assert_array_equal(recovered[0], np.array([5.0], dtype=np.float32))
