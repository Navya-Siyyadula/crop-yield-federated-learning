"""
Model comparison pipeline for crop yield prediction.

Validation set:
    Used for model comparison.

Test set:
    Reserved for final evaluation after selecting the model.
"""

import numpy as np

from src.models.baseline import (
    load_processed_data,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
)
from src.models.evaluate import evaluate_model


def mean_predictor(y_train, y_val):
    """Evaluate a simple baseline that always predicts train mean."""

    prediction = np.full(len(y_val), y_train.mean())

    from src.models.evaluate import calculate_metrics

    return calculate_metrics(y_val, prediction)


def print_results(name, metrics):
    print(
        f"{name:<20} | "
        f"MAE: {metrics['MAE']:.4f} | "
        f"RMSE: {metrics['RMSE']:.4f} | "
        f"R²: {metrics['R2']:.4f}"
    )


def main():

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_processed_data()

    print("=" * 75)
    print("CROP YIELD MODEL COMPARISON")
    print("=" * 75)

    print(f"Training data   : {X_train.shape}")
    print(f"Validation data : {X_val.shape}")
    print(f"Test data       : {X_test.shape}")

    results = {}
    models = {}

    print("\nTraining Mean Predictor...")
    results["Mean Predictor"] = mean_predictor(y_train, y_val)

    print("Training Linear Regression...")
    models["Linear Regression"] = train_linear_regression(
        X_train, y_train
    )
    results["Linear Regression"] = evaluate_model(
        models["Linear Regression"], X_val, y_val
    )

    print("Training Random Forest...")
    models["Random Forest"] = train_random_forest(
        X_train, y_train
    )
    results["Random Forest"] = evaluate_model(
        models["Random Forest"], X_val, y_val
    )

    print("Training XGBoost...")
    models["XGBoost"] = train_xgboost(
        X_train, y_train
    )
    results["XGBoost"] = evaluate_model(
        models["XGBoost"], X_val, y_val
    )

    print("Training LightGBM...")
    models["LightGBM"] = train_lightgbm(
        X_train, y_train
    )
    results["LightGBM"] = evaluate_model(
        models["LightGBM"], X_val, y_val
    )

    print("\n" + "=" * 75)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 75)

    for name, metrics in results.items():
        print_results(name, metrics)


if __name__ == "__main__":
    main()