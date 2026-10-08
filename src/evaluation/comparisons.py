"""
Architecture comparison tables and visualization generator.

Generates the required final architecture comparison outputs:
- results/final_architecture_comparison.csv
- results/final_latency_comparison.csv
- results/final_energy_comparison.csv
- results/figure_a_latency_comparison.png
- results/figure_b_energy_comparison.png
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.performance.energy import ArchitectureEnergyResult
from src.performance.latency import ArchitectureLatencyResult


FRAMEWORK_LABELS = [
    "Proposed FLyer / EMO-FL",
    "Edge-Cloud without FL",
    "Cloud-only",
]


def validate_architecture_ordering(
    latency_dict: Dict[str, float],
    energy_dict: Dict[str, float],
) -> bool:
    """
    Validate that:
    proposed_latency < edge_cloud_latency < cloud_only_latency
    AND
    proposed_energy < edge_cloud_energy < cloud_only_energy
    """
    p_lat = latency_dict["Proposed FLyer / EMO-FL"]
    e_lat = latency_dict["Edge-Cloud without FL"]
    c_lat = latency_dict["Cloud-only"]

    p_eng = energy_dict["Proposed FLyer / EMO-FL"]
    e_eng = energy_dict["Edge-Cloud without FL"]
    c_eng = energy_dict["Cloud-only"]

    lat_valid = p_lat < e_lat < c_lat
    eng_valid = p_eng < e_eng < c_eng

    if not lat_valid:
        raise ValueError(
            f"Latency ordering violated: Proposed ({p_lat:.2f}s), "
            f"Edge-Cloud ({e_lat:.2f}s), Cloud-only ({c_lat:.2f}s)"
        )
    if not eng_valid:
        raise ValueError(
            f"Energy ordering violated: Proposed ({p_eng:.2f}J), "
            f"Edge-Cloud ({e_eng:.2f}J), Cloud-only ({c_eng:.2f}J)"
        )

    return True


def save_final_architecture_csvs(
    latency_results: Dict[str, ArchitectureLatencyResult],
    energy_results: Dict[str, ArchitectureEnergyResult],
    results_dir: str | Path = "results",
) -> Tuple[Path, Path, Path]:
    """
    Save the primary and detailed architecture comparison CSVs.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_map = {k: v.total_latency_seconds for k, v in latency_results.items()}
    eng_map = {k: v.total_energy_joules for k, v in energy_results.items()}

    # Validate ordering
    validate_architecture_ordering(lat_map, eng_map)

    # 1. Primary final_architecture_comparison.csv
    primary_csv = out_dir / "final_architecture_comparison.csv"
    with open(primary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["framework", "latency_seconds", "energy_joules"])
        for name in FRAMEWORK_LABELS:
            writer.writerow([name, lat_map[name], eng_map[name]])

    # 2. Detailed final_latency_comparison.csv
    latency_csv = out_dir / "final_latency_comparison.csv"
    with open(latency_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "framework",
            "latency_seconds",
            "compute_latency_seconds",
            "communication_latency_seconds",
        ])
        for name in FRAMEWORK_LABELS:
            res = latency_results[name]
            writer.writerow([
                name,
                res.total_latency_seconds,
                res.compute_latency_seconds,
                res.communication_latency_seconds,
            ])

    # 3. Detailed final_energy_comparison.csv
    energy_csv = out_dir / "final_energy_comparison.csv"
    with open(energy_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "framework",
            "energy_joules",
            "energy_kj",
            "edge_energy_joules",
            "cloud_energy_joules",
            "network_energy_joules",
        ])
        for name in FRAMEWORK_LABELS:
            res = energy_results[name]
            writer.writerow([
                name,
                res.total_energy_joules,
                res.total_energy_kj,
                res.edge_energy_joules,
                res.cloud_energy_joules,
                res.network_energy_joules,
            ])

    return primary_csv, latency_csv, energy_csv


def plot_final_graphs(
    latency_results: Dict[str, ArchitectureLatencyResult],
    energy_results: Dict[str, ArchitectureEnergyResult],
    results_dir: str | Path = "results",
) -> Tuple[Path, Path]:
    """
    Generate Figure A (Latency Comparison) and Figure B (Energy Consumption Comparison).
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = FRAMEWORK_LABELS
    latencies = [latency_results[name].total_latency_seconds for name in names]
    energies_kj = [energy_results[name].total_energy_kj for name in names]

    # Elegant, accessible academic color palette
    colors = ["#10b981", "#f59e0b", "#64748b"]  # Emerald (Proposed), Amber (Edge-Cloud), Slate (Cloud-only)
    edge_colors = ["#047857", "#d97706", "#334155"]

    # -------------------------------------------------------------
    # Figure A — Latency Comparison
    # -------------------------------------------------------------
    fig_a, ax_a = plt.subplots(figsize=(8, 6), dpi=300)
    bars_a = ax_a.bar(names, latencies, color=colors, edgecolor=edge_colors, width=0.55, linewidth=1.5, zorder=3)

    ax_a.set_title("Figure A: System Latency Comparison", fontsize=15, fontweight="bold", pad=15)
    ax_a.set_ylabel("Latency (seconds)", fontsize=13, fontweight="bold", labelpad=10)
    ax_a.set_ylim(0, max(latencies) * 1.25)
    ax_a.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Annotate bars with exact values and relative reduction
    base_lat = latencies[2]  # Cloud-only
    for bar, val in zip(bars_a, latencies):
        h = bar.get_height()
        reduction_txt = ""
        if val < base_lat:
            pct = ((base_lat - val) / base_lat) * 100.0
            reduction_txt = f"\n(-{pct:.1f}%)"
        ax_a.annotate(
            f"{val:.2f} s{reduction_txt}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax_a.tick_params(axis="x", labelsize=11)
    ax_a.tick_params(axis="y", labelsize=11)
    plt.tight_layout()

    fig_a_path = out_dir / "figure_a_latency_comparison.png"
    fig_a.savefig(fig_a_path, bbox_inches="tight")
    # Save alias copy
    fig_a.savefig(out_dir / "final_latency_comparison.png", bbox_inches="tight")
    plt.close(fig_a)

    # -------------------------------------------------------------
    # Figure B — Energy Consumption Comparison (kJ)
    # -------------------------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=(8, 6), dpi=300)
    bars_b = ax_b.bar(names, energies_kj, color=colors, edgecolor=edge_colors, width=0.55, linewidth=1.5, zorder=3)

    ax_b.set_title("Figure B: Energy Consumption Comparison", fontsize=15, fontweight="bold", pad=15)
    ax_b.set_ylabel("Energy Consumption (kJ)", fontsize=13, fontweight="bold", labelpad=10)
    ax_b.set_ylim(0, max(energies_kj) * 1.25)
    ax_b.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Annotate bars with exact values and relative reduction
    base_eng = energies_kj[2]  # Cloud-only
    for bar, val in zip(bars_b, energies_kj):
        h = bar.get_height()
        reduction_txt = ""
        if val < base_eng:
            pct = ((base_eng - val) / base_eng) * 100.0
            reduction_txt = f"\n(-{pct:.1f}%)"
        ax_b.annotate(
            f"{val:.2f} kJ{reduction_txt}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax_b.tick_params(axis="x", labelsize=11)
    ax_b.tick_params(axis="y", labelsize=11)
    plt.tight_layout()

    fig_b_path = out_dir / "figure_b_energy_comparison.png"
    fig_b.savefig(fig_b_path, bbox_inches="tight")
    # Save alias copy
    fig_b.savefig(out_dir / "final_energy_comparison.png", bbox_inches="tight")
    plt.close(fig_b)

    return fig_a_path, fig_b_path
