import numpy as np
import inspect
from sklearn.preprocessing import StandardScaler

from src.evaluation.target_scaling import (
    best_validation_epoch,
    fit_target_scaler,
    inverse_transform_targets,
    original_unit_metrics,
    transform_targets,
)


def test_target_scaler_is_fitted_only_on_training_targets():
    y_train = np.array([10.0, 20.0, 30.0])
    scaler = fit_target_scaler(y_train)
    assert isinstance(scaler, StandardScaler)
    assert scaler.n_samples_seen_ == len(y_train)
    assert scaler.mean_[0] == np.mean(y_train)
    assert scaler.var_[0] == np.var(y_train)


def test_validation_and_test_transform_do_not_refit_training_scaler():
    scaler = fit_target_scaler(np.array([10.0, 20.0, 30.0]))
    original_mean = scaler.mean_.copy()
    original_var = scaler.var_.copy()
    val_scaled = transform_targets(scaler, np.array([100.0, 200.0]))
    test_scaled = transform_targets(scaler, np.array([-50.0, 500.0]))
    assert val_scaled.shape == (2,)
    assert test_scaled.shape == (2,)
    np.testing.assert_array_equal(scaler.mean_, original_mean)
    np.testing.assert_array_equal(scaler.var_, original_var)
    assert scaler.n_samples_seen_ == 3


def test_target_scaling_inverse_transform_recovers_original_values():
    y = np.array([2023.56, 4099.09, 5998.29])
    scaler = fit_target_scaler(y)
    recovered = inverse_transform_targets(scaler, transform_targets(scaler, y))
    np.testing.assert_allclose(recovered, y, rtol=1e-6, atol=1e-4)


def test_metrics_are_computed_on_original_target_units():
    actual = np.array([1000.0, 2000.0, 3000.0])
    predicted = np.array([1100.0, 1900.0, 2800.0])
    scores = original_unit_metrics(actual, predicted)
    assert np.isclose(scores["mae"], 400.0 / 3.0)
    assert np.isclose(scores["rmse"], np.sqrt((100.0**2 + 100.0**2 + 200.0**2) / 3))
    assert np.isclose(scores["r2"], 0.97)


def test_best_checkpoint_uses_minimum_validation_loss_only():
    validation_losses = [1.02, 0.91, 0.87, 0.93, 1.14]
    assert len(inspect.signature(best_validation_epoch).parameters) == 1
    assert best_validation_epoch(validation_losses) == (3, 0.87)


def test_best_checkpoint_rejects_empty_or_nonfinite_validation_history():
    import pytest

    with pytest.raises(ValueError, match="(?i)at least one"):
        best_validation_epoch([])
    with pytest.raises(ValueError, match="finite"):
        best_validation_epoch([0.5, float("nan")])
