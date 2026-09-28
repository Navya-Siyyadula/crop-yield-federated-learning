from pathlib import Path
import joblib
import numpy as np
import pandas as pd


EXPECTED_FEATURE_COUNT = 38


def load_preprocessor():
    """Load the frozen fitted preprocessing object."""
    project_root = Path(__file__).resolve().parents[2]
    preprocessor_path = project_root / "data" / "processed" / "preprocessor.pkl"

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at: {preprocessor_path}"
        )

    return joblib.load(preprocessor_path)


def prepare_raw_data(data):
    """Prepare raw input data for the frozen preprocessor."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    data = data.copy()

    # Convert date columns
    for column in ["sowing_date", "harvest_date", "timestamp"]:
        data[column] = pd.to_datetime(data[column])

    # Create the exact date-derived features expected by the frozen preprocessor
    data["sowing_month"] = data["sowing_date"].dt.month
    data["sowing_dayofyear"] = data["sowing_date"].dt.dayofyear

    data["harvest_month"] = data["harvest_date"].dt.month
    data["harvest_dayofyear"] = data["harvest_date"].dt.dayofyear

    data["timestamp_month"] = data["timestamp"].dt.month
    data["timestamp_dayofyear"] = data["timestamp"].dt.dayofyear

    return data


def preprocess(data):
    """
    Convert raw 22-column input into exactly 38 processed features.

    Flow:
    Raw data -> date feature creation -> frozen preprocessor -> 38 features
    """

    data = prepare_raw_data(data)

    preprocessor = load_preprocessor()

    transformed = preprocessor.transform(data)
    transformed = np.asarray(transformed)

    if transformed.ndim != 2:
        raise ValueError(
            f"Expected 2D processed data, got shape {transformed.shape}"
        )

    if transformed.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} processed features, "
            f"but got {transformed.shape[1]}"
        )

    return transformed
