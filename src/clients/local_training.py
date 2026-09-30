import numpy as np
import tensorflow as tf

from src.ml.lstm_model import build_lstm_model


def train_local_model(
    X,
    y,
    initial_weights=None,
    epochs=1,
    batch_size=16
):
    """
    Train one client's LSTM model locally.

    Parameters:
        X: Client feature data, shape (n_samples, 38)
        y: Client target values
        initial_weights: Optional global model weights
        epochs: Number of local training epochs
        batch_size: Local batch size

    Returns:
        trained_weights: Updated model weights
        num_samples: Number of samples used by the client
    """

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if X.ndim != 2 or X.shape[1] != 38:
        raise ValueError(
            f"Expected X shape (n_samples, 38), got {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"X and y sample counts do not match: {len(X)} vs {len(y)}"
        )

    # LSTM expects: (samples, timesteps, features)
    X_lstm = X.reshape(X.shape[0], 1, X.shape[1])

    model = build_lstm_model(input_features=38)

    if initial_weights is not None:
        model.set_weights(initial_weights)

    model.fit(
        X_lstm,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    return model.get_weights(), len(X)