"""Start one encrypted LSTM Flower client against the existing server."""

from __future__ import annotations

import argparse

from flwr.client import start_client

from src.clients.client import EncryptedLSTMFlowerClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-id", required=True, help="Assigned ID, for example client_01")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--data-root", default="data/client_data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    client = EncryptedLSTMFlowerClient(
        client_id=args.client_id,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"Starting Flower client {args.client_id} with {len(client.y_train)} training samples")
    start_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
