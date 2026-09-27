import numpy as np

from sklearn.ensemble import RandomForestRegressor

from src.security.encrypted_update import (
    create_encryption_key,
    serialize_and_encrypt_model,
    decrypt_and_deserialize_model,
)


def test_encrypted_rf_model_roundtrip():

    X = np.random.RandomState(42).rand(30, 4)
    y = np.random.RandomState(42).rand(30)

    model = RandomForestRegressor(
        n_estimators=5,
        random_state=42,
    )

    model.fit(X, y)

    key = create_encryption_key()

    encrypted = serialize_and_encrypt_model(
        model,
        key,
    )

    restored = decrypt_and_deserialize_model(
        encrypted,
        key,
    )

    assert len(key) == 32
    assert len(encrypted) > 0
    assert len(restored.estimators_) == 5

    original_predictions = model.predict(X[:2])
    restored_predictions = restored.predict(X[:2])

    np.testing.assert_allclose(
        original_predictions,
        restored_predictions,
    )


def test_encrypted_update_cannot_use_wrong_key():

    X = np.random.RandomState(42).rand(20, 4)
    y = np.random.RandomState(42).rand(20)

    model = RandomForestRegressor(
        n_estimators=3,
        random_state=42,
    )

    model.fit(X, y)

    key = create_encryption_key()
    wrong_key = create_encryption_key()

    encrypted = serialize_and_encrypt_model(
        model,
        key,
    )

    try:
        decrypt_and_deserialize_model(
            encrypted,
            wrong_key,
        )
        assert False, "Wrong key should fail"
    except Exception:
        pass