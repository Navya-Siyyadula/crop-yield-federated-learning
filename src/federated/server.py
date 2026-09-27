"""
Flower FL server entry point.
"""

import flwr as fl
from src.federated.server_strategy import RFTreeAggregationStrategy


if __name__ == "__main__":
    strategy = RFTreeAggregationStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )