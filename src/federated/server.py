"""
Flower FL server entry point for encrypted LSTM FedAvg.
"""

import flwr as fl

from src.federated.server_strategy import LSTMEncryptedFedAvgStrategy


def weighted_average(metrics):
    total_examples = sum(
        num_examples for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    aggregated = {}

    for num_examples, client_metrics in metrics:
        for key, value in client_metrics.items():
            aggregated[key] = (
                aggregated.get(key, 0.0)
                + float(value) * num_examples
            ) / total_examples

    return aggregated


if __name__ == "__main__":

    strategy = LSTMEncryptedFedAvgStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )