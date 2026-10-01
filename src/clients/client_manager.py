from pathlib import Path

from src.clients.client import EdgeClient


def create_clients(data_dir="data/client_data"):
    """Create the four simulated edge clients.

    Each client is expected to have CSV files in a folder like
    data/client_data/client_01/client_01_X.csv and
    data/client_data/client_01/client_01_y.csv.
    """
    data_dir = Path(data_dir)
    clients = []

    for client_number in range(1, 5):
        client_dir = data_dir / f"client_{client_number:02d}"

        x_path = client_dir / f"client_{client_number}_X.csv"
        y_path = client_dir / f"client_{client_number}_y.csv"

        if not x_path.exists():
            raise FileNotFoundError(f"Missing client feature file: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing client target file: {y_path}")

        clients.append(
            EdgeClient(
                client_id=f"Client{client_number}",
                x_path=x_path,
                y_path=y_path,
            )
        )

    return clients
