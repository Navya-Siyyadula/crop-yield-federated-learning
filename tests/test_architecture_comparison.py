"""
Unit tests for the three-architecture comparison, ordering validation, and output artifacts.
"""

import csv
import json
import tempfile
from pathlib import Path
import pytest

from src.performance.latency import ArchitectureLatencyResult
from src.performance.energy import ArchitectureEnergyResult
from src.performance.emo_optimizer import FLConfigCandidate
from src.evaluation.comparisons import (
    save_final_architecture_csvs,
    validate_architecture_ordering,
    plot_final_graphs,
    FRAMEWORK_LABELS,
)
from src.evaluation.report import (
    generate_final_experiment_config,
    generate_report_ready_results,
)


@pytest.fixture
def sample_results():
    lat_results = {
        "Proposed FLyer / EMO-FL": ArchitectureLatencyResult(
            framework="Proposed FLyer / EMO-FL",
            total_latency_seconds=12.06,
            compute_latency_seconds=9.86,
            communication_latency_seconds=2.20,
            breakdown={"edge_preprocessing_s": 1.75, "parallel_edge_training_s": 8.0, "encryption_overhead_s": 0.01, "uplink_transmission_s": 1.8, "cloud_aggregation_s": 0.02, "downlink_broadcast_s": 0.4},
        ),
        "Edge-Cloud without FL": ArchitectureLatencyResult(
            framework="Edge-Cloud without FL",
            total_latency_seconds=24.09,
            compute_latency_seconds=15.92,
            communication_latency_seconds=8.17,
            breakdown={"edge_preprocessing_s": 1.75, "feature_data_upload_s": 8.05, "cloud_centralized_training_s": 14.17, "downlink_delivery_s": 0.12},
        ),
        "Cloud-only": ArchitectureLatencyResult(
            framework="Cloud-only",
            total_latency_seconds=33.59,
            compute_latency_seconds=16.62,
            communication_latency_seconds=16.97,
            breakdown={"raw_data_upload_s": 16.85, "cloud_preprocessing_s": 2.45, "cloud_centralized_training_s": 14.17, "downlink_delivery_s": 0.12},
        ),
    }

    eng_results = {
        "Proposed FLyer / EMO-FL": ArchitectureEnergyResult(
            framework="Proposed FLyer / EMO-FL",
            total_energy_joules=5198.83,
            total_energy_kj=5.20,
            edge_energy_joules=1200.0,
            cloud_energy_joules=3800.0,
            network_energy_joules=198.83,
            breakdown={},
        ),
        "Edge-Cloud without FL": ArchitectureEnergyResult(
            framework="Edge-Cloud without FL",
            total_energy_joules=9725.70,
            total_energy_kj=9.73,
            edge_energy_joules=1800.0,
            cloud_energy_joules=7600.0,
            network_energy_joules=325.70,
            breakdown={},
        ),
        "Cloud-only": ArchitectureEnergyResult(
            framework="Cloud-only",
            total_energy_joules=13260.50,
            total_energy_kj=13.26,
            edge_energy_joules=2200.0,
            cloud_energy_joules=10400.0,
            network_energy_joules=660.50,
            breakdown={},
        ),
    }

    selected = FLConfigCandidate(
        solution_id=1,
        rounds=2,
        local_epochs=1,
        batch_size=8,
        latency_seconds=12.06,
        energy_joules=5198.83,
        energy_kj=5.20,
        is_selected=True,
    )

    return lat_results, eng_results, selected


def test_ordering_validation_passes(sample_results):
    lat_results, eng_results, _ = sample_results
    lat_map = {k: v.total_latency_seconds for k, v in lat_results.items()}
    eng_map = {k: v.total_energy_joules for k, v in eng_results.items()}

    assert validate_architecture_ordering(lat_map, eng_map) is True


def test_ordering_validation_fails_on_inverted_latency():
    lat_map = {
        "Proposed FLyer / EMO-FL": 30.0,
        "Edge-Cloud without FL": 20.0,
        "Cloud-only": 10.0,
    }
    eng_map = {
        "Proposed FLyer / EMO-FL": 5000.0,
        "Edge-Cloud without FL": 9000.0,
        "Cloud-only": 13000.0,
    }
    with pytest.raises(ValueError, match="Latency ordering violated"):
        validate_architecture_ordering(lat_map, eng_map)


def test_primary_architecture_csv_schema_and_content(sample_results):
    lat_results, eng_results, _ = sample_results

    with tempfile.TemporaryDirectory() as tmp_dir:
        primary_csv, lat_csv, eng_csv = save_final_architecture_csvs(
            lat_results, eng_results, results_dir=tmp_dir
        )

        assert primary_csv.exists()
        assert lat_csv.exists()
        assert eng_csv.exists()

        # Check primary CSV
        with open(primary_csv, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        # Check header
        assert reader[0] == ["framework", "latency_seconds", "energy_joules"]

        # Check exactly 3 rows
        assert len(reader) == 4
        rows = reader[1:]
        framework_names = [r[0] for r in rows]
        assert framework_names == FRAMEWORK_LABELS

        # Check values strictly ascending
        lats = [float(r[1]) for r in rows]
        engs = [float(r[2]) for r in rows]
        assert lats[0] < lats[1] < lats[2]
        assert engs[0] < engs[1] < engs[2]


def test_plot_final_graphs_generation(sample_results):
    lat_results, eng_results, _ = sample_results

    with tempfile.TemporaryDirectory() as tmp_dir:
        fig_a, fig_b = plot_final_graphs(lat_results, eng_results, results_dir=tmp_dir)

        assert fig_a.exists()
        assert fig_b.exists()
        assert fig_a.stat().st_size > 0
        assert fig_b.stat().st_size > 0


def test_report_and_config_generation(sample_results):
    lat_results, eng_results, selected = sample_results

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = generate_final_experiment_config(
            selected, {"num_clients": 4}, results_dir=tmp_dir
        )
        json_path, txt_path = generate_report_ready_results(
            lat_results, eng_results, selected, results_dir=tmp_dir
        )

        assert cfg_path.exists()
        assert json_path.exists()
        assert txt_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["validation_passed"] is True
