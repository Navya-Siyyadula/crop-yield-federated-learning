"""
Federated Averaging (FedAvg).

Combines model updates from multiple clients using
sample-count weighted averaging.
"""

from typing import List, Sequence, Tuple

import numpy as np


ModelWeights = List[np.ndarray]
ClientUpdate = Tuple[ModelWeights, int]


def fedavg(
    client_updates: Sequence[ClientUpdate]
) -> ModelWeights:
    """
    Perform sample-count weighted FedAvg.

    Each client update contains:

        (model_weights, number_of_samples)

    Parameters
    ----------
    client_updates:
        Model weights and sample count from each client.

    Returns
    -------
    ModelWeights:
        Aggregated global model weights.
    """

    if not client_updates:
        raise ValueError(
            "At least one client update is required."
        )

    total_samples = sum(
        sample_count
        for _, sample_count in client_updates
    )

    if total_samples <= 0:
        raise ValueError(
            "Total number of samples must be greater than zero."
        )

    reference_weights = client_updates[0][0]

    if not reference_weights:
        raise ValueError(
            "Model weights cannot be empty."
        )

    # Verify that all clients have the same model structure.
    for weights, sample_count in client_updates:

        if sample_count < 0:
            raise ValueError(
                "Sample count cannot be negative."
            )

        if len(weights) != len(reference_weights):
            raise ValueError(
                "All clients must have the same number of layers."
            )

        for client_layer, reference_layer in zip(
            weights,
            reference_weights
        ):
            if np.asarray(client_layer).shape != np.asarray(
                reference_layer
            ).shape:
                raise ValueError(
                    "Corresponding model layers must have "
                    "the same shape."
                )

    # Initialize global weights with zeros.
    global_weights = [
        np.zeros_like(
            np.asarray(layer),
            dtype=np.float64
        )
        for layer in reference_weights
    ]

    # Weighted averaging.
    for client_weights, sample_count in client_updates:

        client_weight = (
            sample_count / total_samples
        )

        for layer_index, layer in enumerate(
            client_weights
        ):
            global_weights[layer_index] += (
                np.asarray(layer, dtype=np.float64)
                * client_weight
            )

    return [
        np.asarray(layer)
        for layer in global_weights
    ]