import numpy as np

def fedavg(updates):
    """
    updates = [
        (weights, num_samples),
        (weights, num_samples)
    ]
    """

    total_samples = sum(num_samples for _, num_samples in updates)

    num_layers = len(updates[0][0])

    averaged_weights = []

    for layer in range(num_layers):
        weighted_sum = None

        for weights, num_samples in updates:
            layer_weights = weights[layer] * num_samples

            if weighted_sum is None:
                weighted_sum = layer_weights
            else:
                weighted_sum += layer_weights

        averaged_layer = weighted_sum / total_samples
        averaged_weights.append(averaged_layer)

    return averaged_weights
