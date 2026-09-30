"""Append structured performance records to a dedicated CSV artifact."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PERFORMANCE_FIELDS = (
    "timestamp", "experiment_id", "round", "client_id", "number_of_clients",
    "number_of_samples", "training_latency_ms", "serialization_latency_ms",
    "deserialization_latency_ms",
    "encryption_latency_ms", "decryption_latency_ms", "aggregation_latency_ms",
    "encrypted_update_size_bytes",
    "communication_latency_ms", "total_round_latency_ms", "security_overhead_ms",
    "energy_joules",
    "energy_status", "energy_method", "ELEI", "energy_reference_joules",
    "latency_reference_ms", "reference_source",
)


class MetricsLogger:
    """Append rows without touching other result artifacts."""

    def __init__(self, path: str | Path = "results/performance_metrics.csv", experiment_id: str | None = None):
        self.path = Path(path)
        self.experiment_id = experiment_id or uuid4().hex

    def log(self, metrics: dict) -> Path:
        row = {field: None for field in PERFORMANCE_FIELDS}
        row.update(metrics)
        row["timestamp"] = row.get("timestamp") or datetime.now(timezone.utc).isoformat()
        row["experiment_id"] = row.get("experiment_id") or self.experiment_id

        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), [])
            if tuple(header) != PERFORMANCE_FIELDS:
                raise ValueError(f"Unexpected columns in performance log: {self.path}")

        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PERFORMANCE_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        return self.path
