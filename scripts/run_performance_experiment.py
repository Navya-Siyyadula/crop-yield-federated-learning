"""Run the repository's encrypted Flower strategy with compatible real clients."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from uuid import uuid4

import flwr as fl

from src.federated.server_strategy import LSTMEncryptedFedAvgStrategy
from src.performance.metrics_logger import MetricsLogger


class ClientMetricsStrategy(LSTMEncryptedFedAvgStrategy):
    """Add per-client FitRes metrics to the log after the existing aggregation."""

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        for _, fit_res in results:
            client_metrics = fit_res.metrics or {}
            self.metrics_logger.log({
                "round": server_round,
                "client_id": client_metrics.get("client_id"),
                "number_of_clients": len(results),
                "number_of_samples": fit_res.num_examples,
                "training_latency_ms": client_metrics.get("training_latency_ms"),
                "serialization_latency_ms": client_metrics.get("serialization_latency_ms"),
                "encryption_latency_ms": client_metrics.get("encryption_latency_ms"),
                "encrypted_update_size_bytes": client_metrics.get(
                    "encrypted_update_size_bytes"
                ),
            })
        return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start an encrypted LSTM FedAvg Flower server and record per-round "
            "latency/energy metrics. Requires compatible Flower clients."
        )
    )
    parser.add_argument("--address", default="0.0.0.0:8080", help="Flower bind address")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--results", type=Path, default=Path("results/performance_metrics.csv"))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--energy-reference-joules", type=float, default=None)
    parser.add_argument("--latency-reference-ms", type=float, default=None)
    parser.add_argument("--reference-source", default=None)
    parser.add_argument("--w-energy", type=float, default=0.5)
    parser.add_argument("--w-latency", type=float, default=0.5)
    args = parser.parse_args()

    if args.rounds <= 0 or args.min_clients <= 0:
        parser.error("--rounds and --min-clients must be positive")
    has_energy_reference = args.energy_reference_joules is not None
    has_latency_reference = args.latency_reference_ms is not None
    if has_energy_reference != has_latency_reference:
        parser.error("Supply both energy and latency references, or neither")
    if has_energy_reference and not args.reference_source:
        parser.error("--reference-source is required when references are supplied")
    return args


def main() -> None:
    args = parse_args()
    experiment_id = args.experiment_id or uuid4().hex
    logger = MetricsLogger(args.results, experiment_id=experiment_id)
    strategy = ClientMetricsStrategy(
        fraction_fit=1.0,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        metrics_logger=logger,
        energy_reference_joules=args.energy_reference_joules,
        latency_reference_ms=args.latency_reference_ms,
        reference_source=args.reference_source,
        w_energy=args.w_energy,
        w_latency=args.w_latency,
    )

    print(f"Starting encrypted Flower FL experiment {experiment_id}")
    print(f"Waiting for at least {args.min_clients} compatible clients at {args.address}")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    if args.results.exists():
        with args.results.open(newline="", encoding="utf-8") as handle:
            records = [
                row for row in csv.DictReader(handle)
                if row.get("experiment_id") == experiment_id
            ]
    else:
        records = []

    print(f"Completed {len(records)} recorded round(s); metrics: {args.results}")
    for row in records:
        print(
            f"Round {row['round']}: clients={row['number_of_clients']}, "
            f"samples={row['number_of_samples']}, "
            f"round_ms={row['total_round_latency_ms'] or 'unavailable'}, "
            f"energy={row['energy_status']}, ELEI={row['ELEI'] or 'unavailable'}"
        )


if __name__ == "__main__":
    main()
