"""Energy readings and explicit ELEI calculation.

No portable electrical-energy telemetry source is present in this project. The
default provider therefore reports ``unavailable``; it never synthesizes joules.
Callers may supply a measured or externally derived estimated reading explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Literal

EnergyStatus = Literal["measured", "estimated", "unavailable"]


@dataclass(frozen=True)
class EnergyReading:
    energy_joules: float | None
    energy_status: EnergyStatus
    energy_method: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.energy_status not in ("measured", "estimated", "unavailable"):
            raise ValueError("Invalid energy status")
        if not self.energy_method.strip():
            raise ValueError("Energy method must be specified")
        if self.energy_status == "unavailable":
            if self.energy_joules is not None:
                raise ValueError("Unavailable energy must not include a joule value")
        elif (
            self.energy_joules is None
            or not isfinite(float(self.energy_joules))
            or self.energy_joules < 0
        ):
            raise ValueError("Measured or estimated energy requires finite non-negative joules")


def unavailable_energy(method: str = "none", detail: str | None = None) -> EnergyReading:
    return EnergyReading(None, "unavailable", method, detail)


def energy_reading(joules: float, status: Literal["measured", "estimated"], method: str) -> EnergyReading:
    if status not in ("measured", "estimated"):
        raise ValueError("Energy reading status must be measured or estimated")
    value = float(joules)
    if not isfinite(value) or value < 0:
        raise ValueError("Energy must be a finite, non-negative number of joules")
    if not method.strip():
        raise ValueError("Energy method must be specified")
    return EnergyReading(value, status, method)


def calculate_elei(
    energy_joules: float | None,
    latency_ms: float,
    energy_reference_joules: float,
    latency_reference_ms: float,
    *,
    energy_status: EnergyStatus,
    w_energy: float = 0.5,
    w_latency: float = 0.5,
) -> float | None:
    """Compute weighted reference-normalized energy/latency; unavailable energy yields None."""
    if energy_status not in ("measured", "estimated", "unavailable"):
        raise ValueError("energy_status must be measured, estimated, or unavailable")
    values = (latency_ms, energy_reference_joules, latency_reference_ms, w_energy, w_latency)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("ELEI inputs must be finite numbers")
    if energy_reference_joules <= 0 or latency_reference_ms <= 0:
        raise ValueError("ELEI reference values must be greater than zero")
    if latency_ms < 0:
        raise ValueError("Latency must be non-negative")
    if w_energy < 0 or w_latency < 0 or not isclose(w_energy + w_latency, 1.0, abs_tol=1e-9):
        raise ValueError("ELEI weights must be non-negative and sum to 1")
    if energy_status == "unavailable":
        return None
    if energy_joules is None:
        raise ValueError("Measured or estimated energy requires a joule value")
    if not isfinite(float(energy_joules)):
        raise ValueError("Energy must be finite")
    if energy_joules < 0:
        raise ValueError("Energy must be non-negative")
    return w_energy * (energy_joules / energy_reference_joules) + w_latency * (
        latency_ms / latency_reference_ms
    )


class EnergyMeter:
    """Conservative provider shell; no measurement is claimed without a provider."""

    def __init__(self, provider=None, method: str = "none") -> None:
        self._provider = provider
        self._method = method

    def read(self) -> EnergyReading:
        if self._provider is None:
            return unavailable_energy(
                self._method,
                "No supported electrical energy telemetry provider is configured.",
            )
        reading = self._provider()
        if not isinstance(reading, EnergyReading):
            raise TypeError("Energy provider must return an EnergyReading")
        return reading


class CodeCarbonRoundEnergyMeter(EnergyMeter):
    """Estimate whole-machine energy during the active Flower round."""

    METHOD = (
        "CodeCarbon software estimate (CPU/GPU/RAM), whole-machine Flower round scope"
    )

    def __init__(self, tracker_factory=None) -> None:
        super().__init__(method=self.METHOD)
        self._tracker_factory = tracker_factory
        self._tracker = None

    def start_round(self) -> None:
        """Begin an isolated CodeCarbon task at the Flower round boundary."""
        if self._tracker is not None:
            raise RuntimeError("CodeCarbon round measurement is already active")
        if self._tracker_factory is None:
            from codecarbon import EmissionsTracker

            tracker_factory = EmissionsTracker
        else:
            tracker_factory = self._tracker_factory
        self._tracker = tracker_factory(
            save_to_file=False,
            log_level="error",
        )
        self._tracker.start()
        self._tracker.start_task("flower_federated_round")

    def read(self) -> EnergyReading:
        if self._tracker is None:
            return unavailable_energy(self.METHOD, "No Flower round was measured.")

        tracker, self._tracker = self._tracker, None
        try:
            data = tracker.stop_task("flower_federated_round")
            tracker.stop()
        except Exception as exc:
            return unavailable_energy(self.METHOD, f"CodeCarbon failed: {exc}")
        if data is None or data.energy_consumed is None:
            return unavailable_energy(self.METHOD, "CodeCarbon returned no energy estimate.")
        return energy_reading(
            float(data.energy_consumed) * 3_600_000.0,
            "estimated",
            self.METHOD,
        )
