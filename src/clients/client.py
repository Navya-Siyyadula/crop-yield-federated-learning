from pathlib import Path

import pandas as pd

from src.clients.local_training import train_local_model


class EdgeClient:
    """Represents one simulated edge client/farm in the federated pipeline."""

    def __init__(self, client_id, x_path, y_path):
        self.client_id = client_id
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)

    def load_data(self):
        """Load the local client partition from CSV files."""
        X = pd.read_csv(self.x_path).values
        y = pd.read_csv(self.y_path).values.reshape(-1)
        return X, y

    def fit(self, initial_weights=None, epochs=1, batch_size=16):
        """
        Train this client's model locally using its partition.

        Returns
        -------
        tuple
            (updated_weights, num_samples)
        """
        X, y = self.load_data()
        weights, num_samples = train_local_model(
            X=X,
            y=y,
            initial_weights=initial_weights,
            epochs=epochs,
            batch_size=batch_size,
        )
        return weights, num_samples

    def __repr__(self):
        return f"EdgeClient(client_id={self.client_id!r}, x_path={str(self.x_path)!r})"
