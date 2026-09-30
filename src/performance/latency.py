"""Named latency collection for the encrypted federated-learning path."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from src.performance.timer import Timer

LATENCY_STAGES = (
    "local_training_ms",
    "serialization_ms",
    "deserialization_ms",
    "encryption_ms",
    "decryption_ms",
    "aggregation_ms",
    "communication_ms",
    "total_round_ms",
    "security_overhead_ms",
)


class LatencyCollector:
    """Accumulate milliseconds by stage; repeated stage calls are summed."""

    def __init__(self) -> None:
        self._values: dict[str, float] = {}

    def record(self, stage: str, duration_ms: float) -> float:
        if stage not in LATENCY_STAGES:
            raise ValueError(f"Unknown latency stage: {stage}")
        duration = float(duration_ms)
        if duration < 0:
            raise ValueError("Latency cannot be negative")
        self._values[stage] = self._values.get(stage, 0.0) + duration
        return duration

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        timer = Timer()
        with timer:
            yield
        self.record(stage, timer.elapsed_ms or 0.0)

    def get(self, stage: str) -> float | None:
        return self._values.get(stage)

    def stage_totals(self) -> dict[str, float]:
        return dict(self._values)

    def summed_stage_latency_ms(self) -> float:
        """Sum recorded component stages, excluding the round total itself."""
        excluded = {"total_round_ms", "security_overhead_ms"}
        return sum(value for name, value in self._values.items() if name not in excluded)

    def total_latency_ms(self) -> float | None:
        """Return explicitly measured round time, or component sum if absent."""
        return self._values.get("total_round_ms", self.summed_stage_latency_ms() or None)

    def as_dict(self) -> dict[str, float | None]:
        return {stage: self._values.get(stage) for stage in LATENCY_STAGES}
