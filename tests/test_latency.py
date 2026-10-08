"""
Unit tests for system latency evaluation and network modeling.
"""

import pytest

from src.performance.latency import (
    SystemLatencyEvaluator,
    calc_network_latency,
    DEFAULT_UPLINK_BPS,
    DEFAULT_DOWNLINK_BPS,
)


def test_calc_network_latency_calculation():
    # 5 Mbps link = 625,000 bytes/sec
    # 625,000 bytes should take 1.0 second + RTT
    payload = 625000
    rtt = 0.05
    lat = calc_network_latency(payload, DEFAULT_UPLINK_BPS, rtt_seconds=rtt)
    assert pytest.approx(lat, rel=1e-4) == 1.05


def test_calc_network_latency_invalid_inputs():
    with pytest.raises(ValueError):
        calc_network_latency(100, bandwidth_bps=0)
    with pytest.raises(ValueError):
        calc_network_latency(-1, bandwidth_bps=DEFAULT_UPLINK_BPS)


def test_system_latency_ordering():
    evaluator = SystemLatencyEvaluator()
    evaluator.cached_benchmarks = {
        "num_clients": 4,
        "model_serialized_bytes": 105215,
        "model_encrypted_bytes": 105243,
        "encryption_time_s": 0.005,
        "parallel_edge_train_time_per_epoch": 4.0,
        "avg_edge_train_time_per_epoch": 3.8,
        "fedavg_time_s": 0.001,
        "centralized_train_time_per_epoch": 1.7,
        "operational_feature_bytes": 5000000,
        "operational_raw_bytes": 10500000,
        "edge_prep_time_s": 1.75,
        "cloud_prep_time_s": 2.45,
    }

    res_prop = evaluator.evaluate_proposed_flyer(rounds=2, local_epochs=1, batch_size=16)
    res_ec = evaluator.evaluate_edge_cloud_without_fl()
    res_co = evaluator.evaluate_cloud_only()

    assert res_prop.total_latency_seconds < res_ec.total_latency_seconds
    assert res_ec.total_latency_seconds < res_co.total_latency_seconds
    assert res_prop.framework == "Proposed FLyer / EMO-FL"
    assert res_ec.framework == "Edge-Cloud without FL"
    assert res_co.framework == "Cloud-only"
