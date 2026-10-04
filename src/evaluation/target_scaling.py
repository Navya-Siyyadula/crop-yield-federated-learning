"""Train-only target scaling helpers for regression experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def fit_target_scaler(y_train: np.ndarray) -> StandardScaler:
    """Fit a 1D target scaler using training targets only."""
    values = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    if values.shape[0] == 0:
        raise ValueError("Cannot fit target scaler on an empty training target")
    if not np.isfinite(values).all():
        raise ValueError("Training targets must be finite")
    return StandardScaler().fit(values)


def transform_targets(scaler: StandardScaler, targets: np.ndarray) -> np.ndarray:
    """Transform targets with an already-fitted training scaler."""
    values = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(values).all():
        raise ValueError("Targets must be finite")
    return scaler.transform(values).reshape(-1).astype(np.float32)


def inverse_transform_targets(scaler: StandardScaler, targets_scaled: np.ndarray) -> np.ndarray:
    """Return predictions to the original target units."""
    values = np.asarray(targets_scaled, dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(values).all():
        raise ValueError("Scaled predictions must be finite")
    return scaler.inverse_transform(values).reshape(-1)


def original_unit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics after predictions are in original units."""
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted target arrays must have equal shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted targets must be finite")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def best_validation_epoch(validation_losses: np.ndarray | list[float]) -> tuple[int, float]:
    """Return the one-based epoch and loss minimum based only on validation loss."""
    losses = np.asarray(validation_losses, dtype=np.float64).reshape(-1)
    if losses.size == 0:
        raise ValueError("At least one validation loss is required")
    if not np.isfinite(losses).all():
        raise ValueError("Validation losses must be finite")
    index = int(np.argmin(losses))
    return index + 1, float(losses[index])
