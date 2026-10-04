"""Select the best validation-loss checkpoint for the normalized central LSTM."""

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

from src.evaluation.final_experiment import fit_centralized
from src.evaluation.target_scaling import (
    best_validation_epoch,
    fit_target_scaler,
    inverse_transform_targets,
    original_unit_metrics,
    transform_targets,
)


SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
EXPECTED_TARGET = "yield_kg_per_hectare"


def load_split(split: str, rows: int) -> tuple[np.ndarray, np.ndarray]:
    data_dir = ROOT / "data" / "processed"
    x = pd.read_csv(data_dir / f"X_{split}.csv").to_numpy(dtype=np.float32)
    y_frame = pd.read_csv(data_dir / f"y_{split}.csv")
    if x.shape != (rows, 38) or y_frame.shape != (rows, 1):
        raise ValueError(f"Unexpected {split} split shapes: {x.shape}, {y_frame.shape}")
    if y_frame.columns[0] != EXPECTED_TARGET:
        raise ValueError(f"Unexpected target column in {split}: {y_frame.columns[0]}")
    y = y_frame.iloc[:, 0].to_numpy(dtype=np.float32)
    return x.reshape(rows, 1, 38), y


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
    os.environ["PYTHONHASHSEED"] = str(SEED)
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(SEED)

    # Load only training and validation data before checkpoint selection.
    x_train, y_train = load_split("train", 350)
    x_val, y_val = load_split("val", 75)
    scaler = fit_target_scaler(y_train)
    y_train_scaled = transform_targets(scaler, y_train)
    y_val_scaled = transform_targets(scaler, y_val)

    results_dir = ROOT / "results"
    checkpoint_path = results_dir / "lstm_target_scaled_best_validation.weights.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )
    model = fit_centralized(
        x_train, y_train_scaled, x_val, y_val_scaled,
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint],
    )
    history = pd.DataFrame({
        "epoch": np.arange(1, EPOCHS + 1),
        "train_loss": model.history.history["loss"],
        "val_loss": model.history.history["val_loss"],
        "train_mae_scaled": model.history.history["mae"],
        "val_mae_scaled": model.history.history["val_mae"],
    })
    best_epoch, best_loss = best_validation_epoch(history["val_loss"].to_numpy())
    model.load_weights(checkpoint_path)

    # The held-out test split is read only after selecting and restoring weights.
    x_test, y_test = load_split("test", 75)
    pred_scaled = model.predict(x_test, verbose=0).reshape(-1)
    predictions = inverse_transform_targets(scaler, pred_scaled)
    scores = original_unit_metrics(y_test, predictions)

    history.to_csv(results_dir / "lstm_target_scaled_best_checkpoint_history.csv", index=False)
    pd.DataFrame({"actual_yield": y_test, "predicted_yield": predictions}).to_csv(
        results_dir / "lstm_target_scaled_best_checkpoint_predictions.csv", index=False
    )

    previous_path = results_dir / "lstm_target_scaled_results.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    final_epoch_metrics = previous["reported_metrics_kg_per_hectare"]
    final_epoch_prediction_stats = previous["prediction_stats_kg_per_hectare"]
    independent_scores = original_unit_metrics(y_test, predictions)
    result = {
        "experiment": "normalized centralized LSTM, best validation-loss checkpoint",
        "seed": SEED,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "architecture": "Input(1,38) -> LSTM(64) -> Dense(32,relu) -> Dense(1)",
        "input_shape": [1, 38],
        "target_scaler": "StandardScaler",
        "target_scaler_fit_on": "y_train only",
        "target_scaler_mean": float(scaler.mean_[0]),
        "target_scaler_scale": float(scaler.scale_[0]),
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam (existing default)",
        "loss": "MSE (standardized target)",
        "best_epoch": best_epoch,
        "best_validation_loss_scaled_mse": best_loss,
        "validation_losses_by_epoch": history["val_loss"].astype(float).tolist(),
        "test_used_for_checkpoint_selection": False,
        "test_mae_kg_per_hectare": scores["mae"],
        "test_rmse_kg_per_hectare": scores["rmse"],
        "test_r2": scores["r2"],
        "independent_sklearn_metrics_kg_per_hectare": independent_scores,
        "actual_test_stats_kg_per_hectare": stats(y_test),
        "prediction_stats_kg_per_hectare": stats(predictions),
        "final_epoch_normalized_lstm": {
            "selected_epoch": EPOCHS,
            "metrics_kg_per_hectare": final_epoch_metrics,
            "prediction_stats_kg_per_hectare": final_epoch_prediction_stats,
        },
        "checkpoint_weights_file": checkpoint_path.name,
        "training_history_first_epoch": history.iloc[0].to_dict(),
        "training_history_last_epoch": history.iloc[-1].to_dict(),
    }
    (results_dir / "lstm_target_scaled_best_checkpoint_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(history["epoch"], history["train_loss"], label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.scatter([best_epoch], [best_loss], color="red", zorder=3, label=f"Best epoch {best_epoch}")
    ax.axvline(best_epoch, color="red", linestyle=":", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardized target)")
    ax.set_title("Target-normalized LSTM: best validation checkpoint")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "lstm_target_scaled_best_checkpoint_history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.75)
    low = float(min(y_test.min(), predictions.min()))
    high = float(max(y_test.max(), predictions.max()))
    ax.plot([low, high], [low, high], color="red", linestyle="--", label="y = x")
    ax.set_xlabel("Actual yield (kg/ha)")
    ax.set_ylabel("Predicted yield (kg/ha)")
    ax.set_title("Best-validation-checkpoint normalized LSTM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "lstm_target_scaled_best_checkpoint_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    print(json.dumps({
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test_metrics_kg_per_hectare": scores,
        "actual_test_stats": stats(y_test),
        "best_checkpoint_prediction_stats": stats(predictions),
        "final_epoch_prediction_stats": final_epoch_prediction_stats,
        "history_first": history.iloc[0].to_dict(),
        "history_last": history.iloc[-1].to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
