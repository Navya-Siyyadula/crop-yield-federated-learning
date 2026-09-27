import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the already-preprocessed train, validation and test data."""

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def main():
    print("=" * 60)
    print("ML BASELINE - RANDOM FOREST REGRESSION")
    print("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Target:             yield_kg_per_hectare")

    # Create baseline model
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Metrics
    train_metrics = calculate_metrics(y_train, train_pred)
    val_metrics = calculate_metrics(y_val, val_pred)
    test_metrics = calculate_metrics(y_test, test_pred)

    # Display metrics
    print("\nTRAIN METRICS")
    print(f"MAE:  {train_metrics['MAE']:.4f}")
    print(f"RMSE: {train_metrics['RMSE']:.4f}")
    print(f"R²:   {train_metrics['R2']:.4f}")

    print("\nVALIDATION METRICS")
    print(f"MAE:  {val_metrics['MAE']:.4f}")
    print(f"RMSE: {val_metrics['RMSE']:.4f}")
    print(f"R²:   {val_metrics['R2']:.4f}")

    print("\nTEST METRICS")
    print(f"MAE:  {test_metrics['MAE']:.4f}")
    print(f"RMSE: {test_metrics['RMSE']:.4f}")
    print(f"R²:   {test_metrics['R2']:.4f}")

    # Save metrics
    metrics = {
        "model": "RandomForestRegressor",
        "task": "regression",
        "target": "yield_kg_per_hectare",
        "n_features": int(X_train.shape[1]),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = RESULTS_DIR / "baseline_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save trained model
    model_path = RESULTS_DIR / "baseline_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("\n" + "=" * 60)
    print("BASELINE COMPLETE")
    print("=" * 60)
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model saved to:   {model_path}")


if __name__ == "__main__":
    main()