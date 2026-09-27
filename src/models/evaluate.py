"""
Evaluation utilities for crop yield regression models.

Metrics:
    - MAE  : Mean Absolute Error
    - RMSE : Root Mean Squared Error
    - R²   : Coefficient of Determination
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression performance metrics.

    Parameters
    ----------
    y_true : array-like
        Actual crop yield values.

    y_pred : array-like
        Predicted crop yield values.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE and R².
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def evaluate_model(model, X, y):
    """
    Generate predictions and calculate regression metrics.

    Parameters
    ----------
    model : trained model
        A fitted regression model.

    X : array-like
        Input features.

    y : array-like
        Actual target values.

    Returns
    -------
    dict
        MAE, RMSE and R² metrics.
    """

    predictions = model.predict(X)

    return calculate_metrics(y, predictions)