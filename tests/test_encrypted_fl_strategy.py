import numpy as np
from sklearn.ensemble import RandomForestRegressor

from flwr.common import (
    FitRes,
    Parameters,
    Status,
    Code,
)

from src.security.encrypted_update import (
    encrypted_model_to_parameters,
    parameters_to_encrypted_model,
)

from src.security.key_manager import (
    encode_key,
    generate_session_key,
)

from src.federated.server_strategy import (
    RFTreeAggregationStrategy,
)


def test_encrypted_fl_strategy_roundtrip(monkeypatch):

    # Create one shared AES-256 key
    key = generate_session_key()

    monkeypatch.setenv(
        "FL_AES_KEY",
        encode_key(key),
    )

    rng = np.random.RandomState(42)

    # Create synthetic client datasets
    X1 = rng.rand(20, 4)
    y1 = rng.rand(20)

    X2 = rng.rand(20, 4)
    y2 = rng.rand(20)

    # Train client Random Forests
    model1 = RandomForestRegressor(
        n_estimators=5,
        random_state=42,
    ).fit(X1, y1)

    model2 = RandomForestRegressor(
        n_estimators=5,
        random_state=43,
    ).fit(X2, y2)

    # Encrypt both client models
    params1 = encrypted_model_to_parameters(
        model1,
        key,
    )

    params2 = encrypted_model_to_parameters(
        model2,
        key,
    )

    # Create Flower FitRes objects
    status = Status(
        code=Code.OK,
        message="OK",
    )

    fit_res1 = FitRes(
        status=status,
        parameters=params1,
        num_examples=20,
        metrics={},
    )

    fit_res2 = FitRes(
        status=status,
        parameters=params2,
        num_examples=20,
        metrics={},
    )

    # Create the server strategy
    strategy = RFTreeAggregationStrategy()

    # Run the actual server aggregation method
    global_parameters, metrics = strategy.aggregate_fit(
        server_round=1,
        results=[
            (None, fit_res1),
            (None, fit_res2),
        ],
        failures=[],
    )

    assert global_parameters is not None

    # Decrypt the global model
    global_model = parameters_to_encrypted_model(
        global_parameters,
        key,
    )

    # Verify the aggregated model exists
    assert len(global_model.estimators_) > 0

    # Verify prediction works
    predictions = global_model.predict(
        X1[:2]
    )

    assert len(predictions) == 2

    print(
        "Encrypted Flower FL strategy test successful"
    )
    print(
        "Global trees:",
        len(global_model.estimators_),
    )
    print(
        "Predictions:",
        predictions,
    )