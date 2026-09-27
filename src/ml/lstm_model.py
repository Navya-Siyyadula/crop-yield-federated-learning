from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(input_features=38):
    """
    Build the LSTM regression model for crop-yield prediction.

    Input:
        One timestep containing 38 processed features.

    Output:
        Predicted crop yield in kg/hectare.
    """

    model = keras.Sequential([
        layers.Input(shape=(1, input_features)),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model
