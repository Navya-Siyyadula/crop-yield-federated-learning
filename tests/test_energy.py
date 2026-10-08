"""
Unit tests for system energy evaluation and modeling.
"""

import pytest

from src.performance.energy import SystemEnergyEvaluator
from src.performance.latency import ArchitectureLatencyResult


def test_energy_ordering_and_conversion():
    evaluator = SystemEnergyEvaluator()

    lat_prop = ArchitectureLatencyResult(
        framework="Proposed FLyer / EMO-FL",
        total_latency_seconds=12.0,
        compute_latency_seconds=9.8,
        communication_latency_seconds=2.2,
        breakdown={
            "edge_preprocessing_s": 1.75,
            "parallel_edge_training_s": 8.0,
            "encryption_overhead_s": 0.01,
            "uplink_transmission_s": 1.8,
            "cloud_aggregation_s": 0.02,
            "downlink_broadcast_s": 0.4,
        },
    )

    lat_ec = ArchitectureLatencyResult(
        framework="Edge-Cloud without FL",
        total_latency_seconds=24.0,
        compute_latency_seconds=15.8,
        communication_latency_seconds=8.2,
        breakdown={
            "edge_preprocessing_s": 1.75,
            "feature_data_upload_s": 8.0,
            "cloud_centralized_training_s": 14.0,
            "downlink_delivery_s": 0.2,
        },
    )

    lat_co = ArchitectureLatencyResult(
        framework="Cloud-only",
        total_latency_seconds=33.5,
        compute_latency_seconds=16.5,
        communication_latency_seconds=17.0,
        breakdown={
            "raw_data_upload_s": 16.8,
            "cloud_preprocessing_s": 2.5,
            "cloud_centralized_training_s": 14.0,
            "downlink_delivery_s": 0.2,
        },
    )

    eng_prop = evaluator.evaluate_proposed_flyer(lat_prop)
    eng_ec = evaluator.evaluate_edge_cloud_without_fl(lat_ec)
    eng_co = evaluator.evaluate_cloud_only(lat_co)

    # Required ordering
    assert eng_prop.total_energy_joules < eng_ec.total_energy_joules
    assert eng_ec.total_energy_joules < eng_co.total_energy_joules

    # kJ conversion (1 kJ = 1000 J)
    assert pytest.approx(eng_prop.total_energy_kj, rel=1e-3) == eng_prop.total_energy_joules / 1000.0
    assert pytest.approx(eng_ec.total_energy_kj, rel=1e-3) == eng_ec.total_energy_joules / 1000.0
    assert pytest.approx(eng_co.total_energy_kj, rel=1e-3) == eng_co.total_energy_joules / 1000.0
