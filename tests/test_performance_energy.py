import csv

import pytest

from src.performance.energy import (
    EnergyMeter,
    energy_reading,
    calculate_elei,
    unavailable_energy,
)
from src.performance.metrics_logger import MetricsLogger, PERFORMANCE_FIELDS


def test_energy_statuses_and_no_default_fallback():
    assert EnergyMeter().read() == unavailable_energy(
        "none", "No supported electrical energy telemetry provider is configured."
    )
    assert energy_reading(3, "measured", "external meter").energy_status == "measured"
    assert energy_reading(3, "estimated", "explicit calibrated model").energy_joules == 3
    with pytest.raises(ValueError):
        energy_reading(-1, "estimated", "bad")


def test_elei_equal_to_reference_is_one():
    assert calculate_elei(
        10, 100, 10, 100, energy_status="measured", w_energy=0.5, w_latency=0.5
    ) == pytest.approx(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_reference_joules": 0},
        {"energy_reference_joules": -1},
        {"latency_reference_ms": 0},
        {"latency_reference_ms": -1},
        {"w_energy": -0.1, "w_latency": 1.1},
        {"w_energy": 0.4, "w_latency": 0.4},
    ],
)
def test_elei_rejects_invalid_references_and_weights(kwargs):
    args = {
        "energy_reference_joules": 1,
        "latency_reference_ms": 1,
        "energy_status": "estimated",
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        calculate_elei(1, 1, **args)


def test_elei_unavailable_energy_returns_none():
    assert calculate_elei(
        None, 10, 1, 10, energy_status="unavailable"
    ) is None


def test_logger_csv_schema_values_and_preserves_other_results(tmp_path):
    old_result = tmp_path / "encryption_overhead.csv"
    old_result.write_text("existing,artifact\nkeep,this\n", encoding="utf-8")
    output = tmp_path / "performance_metrics.csv"
    logger = MetricsLogger(output, experiment_id="exp-a")
    logger.log({
        "round": 4,
        "number_of_clients": 2,
        "number_of_samples": 50,
        "aggregation_latency_ms": 3.5,
        "energy_joules": None,
        "energy_status": "unavailable",
        "ELEI": None,
    })

    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert tuple(reader.fieldnames) == PERFORMANCE_FIELDS
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "exp-a"
    assert rows[0]["round"] == "4"
    assert rows[0]["aggregation_latency_ms"] == "3.5"
    assert rows[0]["energy_status"] == "unavailable"
    assert rows[0]["energy_joules"] == ""
    assert rows[0]["ELEI"] == ""
    assert old_result.read_text(encoding="utf-8") == "existing,artifact\nkeep,this\n"


def test_logger_appends_without_duplicate_header(tmp_path):
    output = tmp_path / "performance.csv"
    logger = MetricsLogger(output, experiment_id="exp-b")
    logger.log({"round": 1})
    logger.log({"round": 2})
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].split(",") == list(PERFORMANCE_FIELDS)
