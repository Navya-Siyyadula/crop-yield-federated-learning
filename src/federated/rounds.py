"""
Multi-round federated learning.
"""

from typing import Callable, List, Sequence, Tuple

from src.federated.fedavg import (
    fedavg,
    ClientUpdate,
    ModelWeights,
)


def run_federated_round(
    global_weights: ModelWeights,
    client_train_functions: Sequence[
        Callable[[ModelWeights], ClientUpdate]
    ],
) -> ModelWeights:

    client_updates: List[ClientUpdate] = []

    for train_client in client_train_functions:

        updated_weights, sample_count = train_client(
            global_weights
        )

        client_updates.append(
            (
                updated_weights,
                sample_count
            )
        )

    return fedavg(client_updates)


def run_federated_training(
    initial_weights: ModelWeights,
    client_train_functions: Sequence[
        Callable[[ModelWeights], ClientUpdate]
    ],
    num_rounds: int = 2,
) -> Tuple[ModelWeights, list]:

    if num_rounds < 1:
        raise ValueError(
            "num_rounds must be at least 1."
        )

    global_weights = initial_weights

    history = []

    for round_number in range(
        1,
        num_rounds + 1
    ):

        global_weights = run_federated_round(
            global_weights,
            client_train_functions
        )

        history.append(
            {
                "round": round_number,
                "global_weights": global_weights,
            }
        )

    return global_weights, history