from federated.fedavg import fedavg


def run_federated_training(global_weights, clients, num_rounds=3):
    print("Starting Federated Learning")

    history = []

    for round_number in range(num_rounds):
        print(f"\nRound {round_number + 1}")

        client_updates = []

        for client in clients:
            print(f"{client} training...")
            update = client(global_weights)
            client_updates.append(update)

        print("Aggregating updates...")
        global_weights = fedavg(client_updates)

        history.append(global_weights)

    print("\nTraining Complete")
    return global_weights, history
