"""
Multi-round federated learning.

Runs multiple rounds of local client training followed by
sample-count weighted Federated Averaging.
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
    """
    Run one round of federated learning.

    Each client receives the current global weights, performs
    local training, and returns updated weights with its
    sample count. FedAvg then creates the new global model.
    """

    if not client_train_functions:
        raise ValueError(
            "At least one client training function is required."
        )

    client_updates: List[ClientUpdate] = []

    for train_client in client_train_functions:

        updated_weights, sample_count = train_client(
            global_weights
        )

        client_updates.append(
            (
                updated_weights,
                sample_count,
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
    """
    Run multi-round Federated Learning.

    Parameters
    ----------
    initial_weights:
        Initial global LSTM model weights.

    client_train_functions:
        Functions representing the federated clients.

    num_rounds:
        Number of FL rounds to execute.

    Returns
    -------
    final_global_weights:
        Global LSTM weights after the final round.

    history:
        List containing the global weights produced after
        each federated round.
    """

    if num_rounds < 1:
        raise ValueError(
            "num_rounds must be at least 1."
        )

    if not client_train_functions:
        raise ValueError(
            "At least one client training function is required."
        )

    global_weights = initial_weights

    history = []

    for round_number in range(1, num_rounds + 1):

        global_weights = run_federated_round(
            global_weights,
            client_train_functions,
        )

        history.append(
            {
                "round": round_number,
                "global_weights": global_weights,
            }
        )

    return global_weights, history