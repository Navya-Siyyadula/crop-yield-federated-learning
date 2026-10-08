"""
Main entry point for running the end-to-end Three-Architecture Comparison Experiment.

Architectures evaluated:
1. Proposed FLyer / EMO-FL (EMO/NSGA-II optimized federated framework)
2. Edge-Cloud framework without FL
3. Cloud-only framework

Outputs generated:
- results/final_architecture_comparison.csv
- results/final_latency_comparison.csv
- results/final_energy_comparison.csv
- results/emo_pareto_solutions.csv
- results/final_experiment_config.json
- results/report_ready_results.txt
- results/report_ready_results.json
- results/figure_a_latency_comparison.png
- results/figure_b_energy_comparison.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.performance.latency import SystemLatencyEvaluator
from src.performance.energy import SystemEnergyEvaluator
from src.performance.emo_optimizer import EMOOptimizer
from src.evaluation.comparisons import (
    validate_architecture_ordering,
    save_final_architecture_csvs,
    plot_final_graphs,
    FRAMEWORK_LABELS,
)
from src.evaluation.report import (
    generate_final_experiment_config,
    generate_report_ready_results,
)


def run_experiment(results_dir: str = "results"):
    print("=" * 80)
    print("RUNNING FINAL THREE-ARCHITECTURE COMPARISON EXPERIMENT")
    print("=" * 80)

    res_path = PROJECT_ROOT / results_dir
    res_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Evaluators with base paper parameters (Uplink 5 Mbps, Downlink 10 Mbps)
    print("\n[Step 1/6] Initializing System Evaluators (Uplink: 5 Mbps, Downlink: 10 Mbps)...")
    latency_evaluator = SystemLatencyEvaluator()
    energy_evaluator = SystemEnergyEvaluator()

    # 2. Run empirical hardware and data benchmarks
    print("\n[Step 2/6] Running empirical benchmarks on client partitions and models...")
    benchmarks = latency_evaluator.run_empirical_benchmarks()
    print(f"  - Serialized model update size: {benchmarks['model_serialized_bytes']:,} bytes")
    print(f"  - AES-256 encrypted update size: {benchmarks['model_encrypted_bytes']:,} bytes")
    print(f"  - Parallel edge train time / epoch: {benchmarks['parallel_edge_train_time_per_epoch']:.3f} s")
    print(f"  - FedAvg cloud aggregation time: {benchmarks['fedavg_time_s'] * 1000:.2f} ms")
    print(f"  - Centralized training time / epoch: {benchmarks['centralized_train_time_per_epoch']:.3f} s")

    # 3. EMO / NSGA-II Optimization for Proposed FLyer
    print("\n[Step 3/6] Running Evolutionary Multi-Objective Optimization (NSGA-II)...")
    emo_optimizer = EMOOptimizer(latency_evaluator, energy_evaluator)
    pareto_front, selected_solution = emo_optimizer.run_optimization()
    pareto_csv_path = res_path / "emo_pareto_solutions.csv"
    emo_optimizer.export_pareto_solutions(pareto_csv_path)

    print(f"  - Evaluated {len(emo_optimizer.population)} candidate FL configurations")
    print(f"  - Found {len(pareto_front)} Pareto-optimal solutions")
    print(f"  - Saved Pareto front to: {pareto_csv_path}")
    print(f"  - Selected EMO-FL Configuration: Solution #{selected_solution.solution_id}")
    print(f"      Rounds: {selected_solution.rounds}, Local Epochs: {selected_solution.local_epochs}, Batch Size: {selected_solution.batch_size}")
    print(f"      Latency: {selected_solution.latency_seconds:.2f} s, Energy: {selected_solution.energy_kj:.2f} kJ ({selected_solution.energy_joules:.2f} J)")

    # 4. Evaluate the Three Architectures
    print("\n[Step 4/6] Evaluating Three System Architectures...")
    lat_proposed = latency_evaluator.evaluate_proposed_flyer(
        rounds=selected_solution.rounds,
        local_epochs=selected_solution.local_epochs,
        batch_size=selected_solution.batch_size,
    )
    energy_proposed = energy_evaluator.evaluate_proposed_flyer(lat_proposed)

    lat_edge_cloud = latency_evaluator.evaluate_edge_cloud_without_fl()
    energy_edge_cloud = energy_evaluator.evaluate_edge_cloud_without_fl(lat_edge_cloud)

    lat_cloud_only = latency_evaluator.evaluate_cloud_only()
    energy_cloud_only = energy_evaluator.evaluate_cloud_only(lat_cloud_only)

    latency_results = {
        "Proposed FLyer / EMO-FL": lat_proposed,
        "Edge-Cloud without FL": lat_edge_cloud,
        "Cloud-only": lat_cloud_only,
    }
    energy_results = {
        "Proposed FLyer / EMO-FL": energy_proposed,
        "Edge-Cloud without FL": energy_edge_cloud,
        "Cloud-only": energy_cloud_only,
    }

    # 5. Mandatory Validation
    print("\n[Step 5/6] Validating Architectural Ordering...")
    lat_map = {k: v.total_latency_seconds for k, v in latency_results.items()}
    eng_map = {k: v.total_energy_joules for k, v in energy_results.items()}
    validate_architecture_ordering(lat_map, eng_map)
    print("  [PASS] proposed_latency < edge_cloud_latency < cloud_only_latency")
    print(f"         {lat_map['Proposed FLyer / EMO-FL']:.2f} s < {lat_map['Edge-Cloud without FL']:.2f} s < {lat_map['Cloud-only']:.2f} s")
    print("  [PASS] proposed_energy < edge_cloud_energy < cloud_only_energy")
    print(f"         {energy_results['Proposed FLyer / EMO-FL'].total_energy_kj:.2f} kJ < {energy_results['Edge-Cloud without FL'].total_energy_kj:.2f} kJ < {energy_results['Cloud-only'].total_energy_kj:.2f} kJ")

    # 6. Export Results and Figures
    print("\n[Step 6/6] Generating Results Files and Plots...")
    primary_csv, latency_csv, energy_csv = save_final_architecture_csvs(
        latency_results,
        energy_results,
        results_dir=res_path,
    )
    print(f"  - Primary Architecture CSV : {primary_csv}")
    print(f"  - Latency Comparison CSV    : {latency_csv}")
    print(f"  - Energy Comparison CSV     : {energy_csv}")

    fig_a, fig_b = plot_final_graphs(
        latency_results,
        energy_results,
        results_dir=res_path,
    )
    print(f"  - Figure A (Latency)       : {fig_a}")
    print(f"  - Figure B (Energy)        : {fig_b}")

    config_path = generate_final_experiment_config(
        selected_solution,
        benchmarks,
        results_dir=res_path,
    )
    json_path, txt_path = generate_report_ready_results(
        latency_results,
        energy_results,
        selected_solution,
        results_dir=res_path,
    )
    print(f"  - Experiment Config JSON   : {config_path}")
    print(f"  - Report Ready Results JSON: {json_path}")
    print(f"  - Report Ready Results TXT : {txt_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT SUCCESSFULLY COMPLETED AND VALIDATED")
    print("=" * 80)
    with open(txt_path, "r", encoding="utf-8") as f:
        print(f.read())

    return {
        "latency_results": latency_results,
        "energy_results": energy_results,
        "selected_solution": selected_solution,
    }


if __name__ == "__main__":
    run_experiment()
