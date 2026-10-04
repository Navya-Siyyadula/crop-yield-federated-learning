"""Run a controlled centralized LSTM experiment with train-only target scaling."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.final_experiment import fit_centralized, load_frozen_data
from src.evaluation.target_scaling import (
    fit_target_scaler,
    inverse_transform_targets,
    original_unit_metrics,
    transform_targets,
)


SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
BASELINE = {
    "mae": 3978.970458984375,
    "rmse": 4160.494441770113,
    "r2": -10.712525367736816,
    "prediction_stats": {
        "mean": 120.1208724975586,
        "std": 3.5787694454193115,
        "min": 108.78119659423828,
        "max": 126.45631408691406,
    },
}


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    # PYTHONHASHSEED is recorded as requested; Python reads hash randomization
    # at interpreter startup, so the explicit RNG calls above seed runtime RNGs.
    os.environ["PYTHONHASHSEED"] = str(SEED)
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(SEED)

    data = load_frozen_data()
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    x_test, y_test = data["test"]

    scaler = fit_target_scaler(y_train)
    y_train_scaled = transform_targets(scaler, y_train)
    y_val_scaled = transform_targets(scaler, y_val)
    # Transform the held-out target only for scale diagnostics; it is not used
    # to fit the scaler or influence training/checkpoint decisions.
    y_test_scaled = transform_targets(scaler, y_test)

    model = fit_centralized(
        x_train, y_train_scaled, x_val, y_val_scaled,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
    )
    predicted_scaled = model.predict(x_test, verbose=0).reshape(-1)
    predicted = inverse_transform_targets(scaler, predicted_scaled)
    metrics = original_unit_metrics(y_test, predicted)

    history = pd.DataFrame({
        "epoch": np.arange(1, EPOCHS + 1),
        "train_loss": model.history.history["loss"],
        "val_loss": model.history.history["val_loss"],
        "train_mae_scaled": model.history.history["mae"],
        "val_mae_scaled": model.history.history["val_mae"],
    })
    results_dir = ROOT / "results"
    history.to_csv(results_dir / "lstm_target_scaled_history.csv", index=False)
    pd.DataFrame({"actual_yield": y_test, "predicted_yield": predicted}).to_csv(
        results_dir / "lstm_target_scaled_predictions.csv", index=False
    )

    target_stats = {
        split: stats(values)
        for split, values in (("train", y_train), ("validation", y_val), ("test", y_test))
    }
    independent_metrics = original_unit_metrics(y_test, predicted)
    result = {
        "experiment": "centralized LSTM with train-only target StandardScaler",
        "seed": SEED,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "target_scaler": type(scaler).__name__,
        "target_scaler_fit_on": "y_train only",
        "target_scaler_mean": float(scaler.mean_[0]),
        "target_scaler_scale": float(scaler.scale_[0]),
        "normalized_target_stats": {
            "train": stats(y_train_scaled),
            "validation": stats(y_val_scaled),
            "test_diagnostic_only": stats(y_test_scaled),
        },
        "target_stats_kg_per_hectare": target_stats,
        "architecture": "Input(1,38) -> LSTM(64) -> Dense(32,relu) -> Dense(1)",
        "input_shape": [1, 38],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam (existing default)",
        "loss": "MSE (on train-standardized target)",
        "training_shuffle": False,
        "prediction_stats_kg_per_hectare": stats(predicted),
        "actual_test_stats_kg_per_hectare": stats(y_test),
        "reported_metrics_kg_per_hectare": metrics,
        "independent_sklearn_metrics_kg_per_hectare": independent_metrics,
        "baseline_raw_target_lstm": BASELINE,
        "training_history_first_epoch": history.iloc[0].to_dict(),
        "training_history_last_epoch": history.iloc[-1].to_dict(),
        "test_used_for_scaler_fit_or_training_decisions": False,
    }
    (results_dir / "lstm_target_scaled_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(y_test, predicted, alpha=0.75)
    low = float(min(y_test.min(), predicted.min()))
    high = float(max(y_test.max(), predicted.max()))
    ax.plot([low, high], [low, high], color="red", linestyle="--", label="y = x")
    ax.set_xlabel("Actual yield (kg/ha)")
    ax.set_ylabel("Predicted yield (kg/ha)")
    ax.set_title("Target-normalized centralized LSTM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "lstm_target_scaled_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(history["epoch"], history["train_loss"], label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardized target)")
    ax.set_title("Target-normalized LSTM training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "lstm_target_scaled_history.png", dpi=160)
    plt.close(fig)

    print(json.dumps({
        "metrics_kg_per_hectare": metrics,
        "actual_test_stats": stats(y_test),
        "prediction_stats": stats(predicted),
        "history_first": history.iloc[0].to_dict(),
        "history_last": history.iloc[-1].to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
