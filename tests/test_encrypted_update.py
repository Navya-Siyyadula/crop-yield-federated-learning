import pytest
def test_wrong_key_rejects_encrypted_parameters():
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    from src.security.encrypted_update import (
        create_encryption_key,
        encrypted_model_to_parameters,
        parameters_to_encrypted_model,
    )

    X = np.random.RandomState(42).rand(20, 4)
    y = np.random.RandomState(42).rand(20)

    model = RandomForestRegressor(
        n_estimators=3,
        random_state=42,
    ).fit(X, y)

    correct_key = create_encryption_key()
    wrong_key = create_encryption_key()

    parameters = encrypted_model_to_parameters(
        model,
        correct_key,
    )

    with pytest.raises(Exception):
        parameters_to_encrypted_model(
            parameters,
            wrong_key,
        )


def test_empty_encrypted_parameters_rejected():
    import numpy as np
    from flwr.common import ndarrays_to_parameters

    from src.security.encrypted_update import (
        create_encryption_key,
        parameters_to_encrypted_model,
    )

    parameters = ndarrays_to_parameters(
        [np.array([], dtype=np.uint8)]
    )

    key = create_encryption_key()

    with pytest.raises(ValueError):
        parameters_to_encrypted_model(
            parameters,
            key,
        )