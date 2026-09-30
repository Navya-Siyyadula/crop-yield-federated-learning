from src.clients.client_manager import create_clients


def get_clients():
    """Return all four simulated edge clients."""

    clients = create_clients()

    print(f"Loaded {len(clients)} edge clients.")

    for client in clients:
        X, y = client.load_data()

        print(
            f"{client.client_id}: "
            f"X={X.shape}, y={y.shape}"
        )

    return clients