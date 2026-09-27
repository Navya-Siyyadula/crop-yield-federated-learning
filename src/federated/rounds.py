import random
from src.federated.fedavg import fedavg

def run_round(global_model, clients):
    print("Running one round...")
    updates = []

    for c in clients:
        print(f"{c} training...")
        update = random.uniform(0.5, 1.5)
        updates.append(update)

    return updates


def run_federated_training(global_model, clients, num_rounds=3):
    print("🚀 Starting Federated Learning")

    for r in range(num_rounds):
        print(f"\n🔄 Round {r+1}")

        updates = run_round(global_model, clients)

        print("Aggregating updates...")
        global_weights = fedavg(updates)

        print("Updating global model...")
        global_model.update(global_weights)

    print("\n✅ Training Complete")
    return global_model
