import random
from federated.fedavg import fedavg

def run_round(global_model, clients):
    print("Running one round...")
    updates = []

    for c in clients:
        updated_weights, num_samples = c(global_model)
        updates.append((updated_weights, num_samples))

    return updates


def run_federated_training(global_model, clients, num_rounds=3):

    history = []

    for _ in range(num_rounds):
        updates = run_round(global_model, clients)

        global_model = fedavg(updates)

        history.append(global_model)

    return global_model, history
