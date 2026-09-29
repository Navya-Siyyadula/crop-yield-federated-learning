import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "lstm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load frozen preprocessed data
X_train = pd.read_csv(DATA_DIR / "X_train.csv").to_numpy(dtype=np.float32)
X_val = pd.read_csv(DATA_DIR / "X_val.csv").to_numpy(dtype=np.float32)
X_test = pd.read_csv(DATA_DIR / "X_test.csv").to_numpy(dtype=np.float32)

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns").to_numpy(dtype=np.float32)
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns").to_numpy(dtype=np.float32)
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns").to_numpy(dtype=np.float32)

# Convert 38 features into one timestep
X_train = X_train.reshape(-1, 1, 38)
X_val = X_val.reshape(-1, 1, 38)
X_test = X_test.reshape(-1, 1, 38)

# Thanisha's exact LSTM architecture
model = keras.Sequential([
    keras.layers.Input(shape=(1, 38)),
    keras.layers.LSTM(64),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print("=" * 70)
print("LSTM TRAINING — FROZEN DATASET")
print("=" * 70)
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

pred = model.predict(X_test, verbose=0).reshape(-1)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n" + "=" * 70)
print("LSTM — FINAL TEST RESULTS")
print("=" * 70)
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

model_path = MODEL_DIR / "lstm_crop_yield_frozen.keras"
model.save(model_path)

print("\nSaved model:", model_path)
