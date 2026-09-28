"""
Federated Averaging (FedAvg).

Combines model updates from multiple clients using
sample-count weighted averaging.

Designed for the final LSTM model, but works with any
model represented as a list of NumPy weight arrays.
"""

from typing import List, Sequence, Tuple

import numpy as np


ModelWeights = List[np.ndarray]
ClientUpdate = Tuple[ModelWeights, int]


def fedavg(client_updates: Sequence[ClientUpdate]) -> ModelWeights:
    """
    Perform sample-count weighted Federated Averaging.

    Parameters
    ----------
    client_updates:
        Sequence of (model_weights, sample_count) tuples.

    Returns
    -------
    ModelWeights:
        The aggregated global model weights.

    Notes
    -----
    The dtype of the first client's weights is preserved.
    For the project's LSTM model, this normally means float32.
    """

    if not client_updates:
        raise ValueError("At least one client update is required.")

    total_samples = sum(
        sample_count for _, sample_count in client_updates
    )

    if total_samples <= 0:
        raise ValueError(
            "Total number of samples must be greater than zero."
        )

    reference_weights = client_updates[0][0]

    if not reference_weights:
        raise ValueError("Model weights cannot be empty.")

    for weights, sample_count in client_updates:

        if sample_count < 0:
            raise ValueError("Sample count cannot be negative.")

        if len(weights) != len(reference_weights):
            raise ValueError(
                "All clients must have the same number of layers."
            )

        for client_layer, reference_layer in zip(
            weights, reference_weights
        ):
            if np.asarray(client_layer).shape != np.asarray(
                reference_layer
            ).shape:
                raise ValueError(
                    "Corresponding model layers must have the same shape."
                )

    global_weights = [
        np.zeros_like(np.asarray(layer))
        for layer in reference_weights
    ]

    for client_weights, sample_count in client_updates:

        client_weight = sample_count / total_samples

        for layer_index, layer in enumerate(client_weights):

            global_weights[layer_index] += (
                np.asarray(layer, dtype=global_weights[layer_index].dtype)
                * client_weight
            )

    return [
        np.asarray(
            layer,
            dtype=np.asarray(reference_weights[i]).dtype
        )
        for i, layer in enumerate(global_weights)
    ]