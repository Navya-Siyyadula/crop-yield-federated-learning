import numpy as np

from src.ml.lstm_model import build_lstm_model


def train_local_model(
    X,
    y,
    initial_weights=None,
    epochs=1,
    batch_size=16,
):
    """
    Train one client's LSTM model locally.

    Parameters
    ----------
    X : array-like
        Client feature data with shape (n_samples, 38).
    y : array-like
        Target values for the client.
    initial_weights : list, optional
        Global model weights to begin local training from.
    epochs : int
        Number of local epochs to run.
    batch_size : int
        Local batch size.

    Returns
    -------
    tuple
        (updated_model_weights, num_samples)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if X.ndim != 2 or X.shape[1] != 38:
        raise ValueError(f"Expected X shape (n_samples, 38), got {X.shape}")

    if len(X) != len(y):
        raise ValueError(f"X and y sample counts do not match: {len(X)} vs {len(y)}")

    X_lstm = X.reshape(X.shape[0], 1, X.shape[1])
    model = build_lstm_model(input_features=38)

    if initial_weights is not None:
        model.set_weights(initial_weights)

    model.fit(
        X_lstm,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    return model.get_weights(), len(X)
