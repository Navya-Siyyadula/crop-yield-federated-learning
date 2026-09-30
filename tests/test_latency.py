import time

import pytest

from src.performance.latency import LatencyCollector
from src.performance.timer import Timer, elapsed_ms


def test_timer_direct_and_context_manager_are_nonnegative(monkeypatch):
    ticks = iter([10.0, 10.25, 20.0, 20.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    timer = Timer(start=True)
    assert timer.stop() == pytest.approx(250.0)
    with Timer() as context_timer:
        pass
    assert context_timer.elapsed_ms == pytest.approx(500.0)


def test_elapsed_ms_helper(monkeypatch):
    monkeypatch.setattr(time, "perf_counter", lambda: 3.125)
    assert elapsed_ms(3.0) == pytest.approx(125.0)


def test_named_measurements_aggregate_and_explicit_total():
    collector = LatencyCollector()
    collector.record("decryption_ms", 2.0)
    collector.record("decryption_ms", 3.0)
    collector.record("aggregation_ms", 7.0)
    assert collector.get("decryption_ms") == 5.0
    assert collector.summed_stage_latency_ms() == 12.0
    assert collector.total_latency_ms() == 12.0
    collector.record("total_round_ms", 20.0)
    assert collector.total_latency_ms() == 20.0
    assert collector.get("communication_ms") is None


def test_latency_rejects_unknown_or_negative_values():
    collector = LatencyCollector()
    with pytest.raises(ValueError):
        collector.record("network_ms", 1)
    with pytest.raises(ValueError):
        collector.record("encryption_ms", -1)
