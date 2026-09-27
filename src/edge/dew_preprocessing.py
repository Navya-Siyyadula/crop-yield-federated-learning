import pandas as pd


def load_client_data(file_path):
    """Load a client's already-preprocessed feature dataset."""
    return pd.read_csv(file_path)


def check_missing_values(data):
    """Check for missing values."""
    missing = data.isnull().sum()

    print("\nMissing values:")
    print(missing)

    total_missing = missing.sum()
    print(f"Total missing values: {total_missing}")

    return total_missing


def check_duplicates(data):
    """Check for duplicate rows."""
    duplicates = data.duplicated().sum()

    print(f"\nDuplicate rows: {duplicates}")

    return duplicates


def check_data_format(data):
    """Check the input format expected by the local ML model."""

    print("\nData shape:", data.shape)
    print("Number of features:", data.shape[1])
    print("Data types:")
    print(data.dtypes.value_counts())


def remove_noise(data):
    """
    Noise validation.

    The client datasets are already normalized and encoded.
    No automatic outlier removal is performed because
    valid standardized values can appear negative or positive.
    """
    return data


def preprocess_client_data(data):
    """
    Dew Computing preprocessing pipeline.

    Normalization and encoding are NOT performed here because
    they were already completed before client splitting.
    """

    print("\n========== DEW PREPROCESSING ==========")

    # 1. Check missing values
    check_missing_values(data)

    # 2. Check duplicate rows
    check_duplicates(data)

    # 3. Check data format
    check_data_format(data)

    # 4. Noise validation
    data = remove_noise(data)

    print("\nDew preprocessing completed.")

    return data


if __name__ == "__main__":

    client_files = [
        "data/clients/client_1_X.csv",
        "data/clients/client_2_X.csv",
        "data/clients/client_3_X.csv",
        "data/clients/client_4_X.csv"
    ]

    for file_path in client_files:

        print("\n\n========================================")
        print(f"Processing: {file_path}")
        print("========================================")

        data = load_client_data(file_path)

        processed_data = preprocess_client_data(data)

        print("\nFinal output shape:", processed_data.shape)
