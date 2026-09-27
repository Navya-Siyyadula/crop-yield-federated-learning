from src.federated.fedavg import fedavg


def run_round(global_model, clients):
    updates = []

    for client in clients:
        updated_weights, num_samples = client(global_model)
        updates.append((updated_weights, num_samples))

    return updates


def run_federated_training(global_model, clients, num_rounds=3):
    history = []

    for _ in range(num_rounds):
        updates = run_round(global_model, clients)

        # aggregate
        global_model = fedavg(updates)

        # save history
        history.append(global_model)

    return global_model, history
