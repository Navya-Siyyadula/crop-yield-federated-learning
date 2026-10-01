from __future__ import annotations

import argparse
from pathlib import Path

from src.clients.client_manager import create_clients


def main():
    parser = argparse.ArgumentParser(
        description="Train the four simulated edge clients locally."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/client_data",
        help="Directory containing the per-client CSV partitions.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of local training epochs per client.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used by each client's local training.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    clients = create_clients(data_dir=str(data_dir))

    print(f"Created {len(clients)} edge clients from {data_dir}")

    for client in clients:
        weights, num_samples = client.fit(epochs=args.epochs, batch_size=args.batch_size)
        print(f"{client.client_id}: trained on {num_samples} samples; weights={len(weights)} tensors")


if __name__ == "__main__":
    main()
