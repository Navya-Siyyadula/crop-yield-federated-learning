import numpy as np

def fedavg(updates):
    """
    updates = [
        ([weights_layer1, weights_layer2, ...], num_samples),
        ...
    ]
    """

    total_samples = sum(num_samples for _, num_samples in updates)

    num_layers = len(updates[0][0])

    aggregated = []

    for layer_idx in range(num_layers):
        weighted_sum = 0

        for weights, num_samples in updates:
            weighted_sum += weights[layer_idx] * num_samples

        aggregated_layer = weighted_sum / total_samples
        aggregated.append(aggregated_layer)

    return aggregated
