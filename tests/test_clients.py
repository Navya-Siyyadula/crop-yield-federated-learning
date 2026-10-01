import numpy as np
import pandas as pd

from src.clients.client import EdgeClient
from src.clients.client_manager import create_clients


def _make_temp_client_data(root_dir, rows=20):
    """Create small synthetic client partitions for testing or demo use."""
    root_dir = root_dir
    for client_num in range(1, 5):
        client_dir = root_dir / f"client_{client_num:02d}"
        client_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42 + client_num)
        X = rng.normal(size=(rows, 38))
        y = 3.0 * X[:, 0] + 2.0 * X[:, 1] - 1.5 * X[:, 2] + client_num

        pd.DataFrame(X).to_csv(client_dir / f"client_{client_num}_X.csv", index=False)
        pd.DataFrame({"yield_kg_per_hectare": y}).to_csv(client_dir / f"client_{client_num}_y.csv", index=False)


def test_edge_client_fit_returns_weights():
    root_dir = __import__("tempfile").TemporaryDirectory()
    base_dir = __import__("pathlib").Path(root_dir.name)
    _make_temp_client_data(base_dir)

    client = EdgeClient(
        client_id="Client1",
        x_path=base_dir / "client_01" / "client_1_X.csv",
        y_path=base_dir / "client_01" / "client_1_y.csv",
    )

    weights, num_samples = client.fit(epochs=1, batch_size=8)
    assert isinstance(weights, list)
    assert num_samples > 0
    assert len(weights) > 0


def test_create_clients_returns_four_clients():
    root_dir = __import__("tempfile").TemporaryDirectory()
    base_dir = __import__("pathlib").Path(root_dir.name)
    _make_temp_client_data(base_dir)

    clients = create_clients(data_dir=str(base_dir))
    assert len(clients) == 4
    assert [client.client_id for client in clients] == ["Client1", "Client2", "Client3", "Client4"]
