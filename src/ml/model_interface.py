from pathlib import Path
import numpy as np
import keras


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "results" / "Sreenidhi_LSTM_best.keras"


def load_model():
    """Load the trained LSTM model."""
    return keras.models.load_model(MODEL_PATH)


def predict(model, features):
    """Predict crop yield from 38 input features."""

    features = np.asarray(features, dtype=np.float32)

    if features.size != 38:
        raise ValueError("Expected exactly 38 input features.")

    features = features.reshape(-1, 1, 38)

    prediction = model.predict(features, verbose=0)

    return prediction.reshape(-1)