# src/federated/server.py

from federated.global_model import GlobalModel
from federated.client_adapter import get_clients
from federated.rounds import run_round
from federated.fedavg import fedavg


def start_federated_training(num_rounds=3):
    print("🚀 Starting Federated Learning Server")

    # Step 1: Load global model
    global_model = GlobalModel()

    # Step 2: Get all clients
    clients = get_clients()

    # Step 3: Run multiple rounds
    for r in range(num_rounds):
        print(f"\n🔄 Round {r+1}")

        # Run one round (clients train + send updates)
        updates = run_round(global_model, clients)

        # Aggregate updates (FedAvg)
        global_weights = fedavg(updates)

        # Update global model
        global_model.update(global_weights)

    print("\n✅ Training Complete")
    return global_model


if __name__ == "__main__":
    start_federated_training()
