from pathlib import Path

import pandas as pd

from src.clients.local_training import train_local_model


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

    def fit(
        self,
        initial_weights=None,
        epochs=1,
        batch_size=16
    ):
        X, y = self.load_data()

        weights, num_samples = train_local_model(
            X=X,
            y=y,
            initial_weights=initial_weights,
            epochs=epochs,
            batch_size=batch_size
        )

        return weights, num_samples