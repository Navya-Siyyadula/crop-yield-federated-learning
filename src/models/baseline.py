"""
Baseline and candidate regression models for crop yield prediction.

Input:
    38 numerical preprocessed features

Target:
    yield_kg_per_hectare
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_processed_data():
    """Load the frozen preprocessed train/validation/test datasets."""

    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "y_train.csv"
    ).squeeze("columns")

    y_val = pd.read_csv(
        PROCESSED_DATA_DIR / "y_val.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        PROCESSED_DATA_DIR / "y_test.csv"
    ).squeeze("columns")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model