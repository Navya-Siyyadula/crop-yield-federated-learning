import numpy as np
import tensorflow as tf

from src.ml.lstm_model import build_lstm_model
from src.clients.client_manager import create_clients
from src.federated.fedavg import fedavg


def main():
    print("=" * 60)
    print("4-CLIENT EDGE INTEGRATION TEST")
    print("=" * 60)

    np.random.seed(42)
    tf.random.set_seed(42)

    # Create the initial global LSTM model
    global_model = build_lstm_model(input_features=38)
    initial_weights = global_model.get_weights()

    print("\nInitial global model:")
    print(f"Weight arrays: {len(initial_weights)}")
    print(f"Weight shapes: {[w.shape for w in initial_weights]}")

    # Load all four clients
    clients = create_clients()

    print(f"\nClients loaded: {len(clients)}")

    client_updates = []

    # Train every client from the same global weights
    for client in clients:
        print(f"\nTraining {client.client_id}...")

        weights, sample_count = client.fit(
            initial_weights=initial_weights,
            epochs=1,
            batch_size=16
        )

        print(f"  Samples: {sample_count}")
        print(f"  Weight arrays: {len(weights)}")
        print(f"  Weight shapes: {[w.shape for w in weights]}")

        # Check weight compatibility
        assert len(weights) == len(initial_weights)

        for client_weight, global_weight in zip(
            weights, initial_weights
        ):
            assert client_weight.shape == global_weight.shape

        client_updates.append((weights, sample_count))

        print("  Status: PASS")

    # Check total samples
    total_samples = sum(
        sample_count for _, sample_count in client_updates
    )

    print(f"\nTotal client samples: {total_samples}")

    assert total_samples == 350

    # Run the project's existing FedAvg
    print("\nRunning FedAvg aggregation...")

    aggregated_weights = fedavg(client_updates)

    # Validate aggregated weights
    assert len(aggregated_weights) == len(initial_weights)

    for aggregated_weight, initial_weight in zip(
        aggregated_weights, initial_weights
    ):
        assert aggregated_weight.shape == initial_weight.shape
        assert np.all(np.isfinite(aggregated_weight))

    print("FedAvg aggregation: PASS")
    print(f"Aggregated weight arrays: {len(aggregated_weights)}")
    print(
        "Aggregated weight shapes:",
        [w.shape for w in aggregated_weights]
    )

    print("\n" + "=" * 60)
    print("OVERALL EDGE INTEGRATION TEST: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()