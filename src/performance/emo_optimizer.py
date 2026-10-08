"""
Evolutionary Multi-Objective Optimization (EMO / NSGA-II) for Federated Learning Configuration.

This module optimizes candidate FL configurations with respect to:
1. Latency (seconds) — Objective 1 (Minimize)
2. Energy Consumption (Joules) — Objective 2 (Minimize)

The selected Pareto-optimal configuration represents the Proposed FLyer / EMO-FL framework.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.performance.energy import SystemEnergyEvaluator
from src.performance.latency import SystemLatencyEvaluator


@dataclass
class FLConfigCandidate:
    """Represents a candidate Federated Learning configuration."""
    solution_id: int
    rounds: int
    local_epochs: int
    batch_size: int
    latency_seconds: float = 0.0
    energy_joules: float = 0.0
    energy_kj: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0
    is_selected: bool = False


class EMOOptimizer:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) optimizer
    for selecting Pareto-optimal FL configurations.
    """

    def __init__(
        self,
        latency_evaluator: SystemLatencyEvaluator,
        energy_evaluator: SystemEnergyEvaluator,
    ):
        self.latency_evaluator = latency_evaluator
        self.energy_evaluator = energy_evaluator
        self.population: List[FLConfigCandidate] = []
        self.pareto_front: List[FLConfigCandidate] = []
        self.selected_solution: Optional[FLConfigCandidate] = None

    def generate_candidate_space(self) -> List[FLConfigCandidate]:
        """Generate candidate FL configurations spanning rounds, epochs, and batch sizes."""
        candidates = []
        sol_id = 1
        rounds_list = [2, 3, 4, 5]
        local_epochs_list = [1, 2, 3, 4]
        batch_sizes = [8, 16, 32]

        for r in rounds_list:
            for e in local_epochs_list:
                for b in batch_sizes:
                    candidates.append(
                        FLConfigCandidate(
                            solution_id=sol_id,
                            rounds=r,
                            local_epochs=e,
                            batch_size=b,
                        )
                    )
                    sol_id += 1

        self.population = candidates
        return candidates

    def evaluate_candidates(self) -> None:
        """Evaluate latency and energy for all candidate configurations."""
        if not self.population:
            self.generate_candidate_space()

        for cand in self.population:
            lat_res = self.latency_evaluator.evaluate_proposed_flyer(
                rounds=cand.rounds,
                local_epochs=cand.local_epochs,
                batch_size=cand.batch_size,
            )
            energy_res = self.energy_evaluator.evaluate_proposed_flyer(lat_res)

            cand.latency_seconds = lat_res.total_latency_seconds
            cand.energy_joules = energy_res.total_energy_joules
            cand.energy_kj = energy_res.total_energy_kj

    def fast_non_dominated_sort(self) -> List[List[FLConfigCandidate]]:
        """
        Fast Non-Dominated Sorting algorithm (Deb et al.).
        Partitions the population into Pareto fronts (F1, F2, ...).
        """
        fronts: List[List[FLConfigCandidate]] = [[]]
        domination_counts: Dict[int, int] = {}
        dominated_solutions: Dict[int, List[FLConfigCandidate]] = {}

        for p in self.population:
            domination_counts[p.solution_id] = 0
            dominated_solutions[p.solution_id] = []

            for q in self.population:
                # Minimization: p dominates q if p is <= in both and < in at least one
                p_dominates_q = (
                    p.latency_seconds <= q.latency_seconds
                    and p.energy_joules <= q.energy_joules
                    and (
                        p.latency_seconds < q.latency_seconds
                        or p.energy_joules < q.energy_joules
                    )
                )
                q_dominates_p = (
                    q.latency_seconds <= p.latency_seconds
                    and q.energy_joules <= p.energy_joules
                    and (
                        q.latency_seconds < p.latency_seconds
                        or q.energy_joules < p.energy_joules
                    )
                )

                if p_dominates_q:
                    dominated_solutions[p.solution_id].append(q)
                elif q_dominates_p:
                    domination_counts[p.solution_id] += 1

            if domination_counts[p.solution_id] == 0:
                p.rank = 1
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front: List[FLConfigCandidate] = []
            for p in fronts[i]:
                for q in dominated_solutions[p.solution_id]:
                    domination_counts[q.solution_id] -= 1
                    if domination_counts[q.solution_id] == 0:
                        q.rank = i + 2
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        self.pareto_front = fronts[0]
        return fronts

    def calculate_crowding_distance(self, front: List[FLConfigCandidate]) -> None:
        """Calculate crowding distances for solutions on a front."""
        n = len(front)
        if n == 0:
            return
        for s in front:
            s.crowding_distance = 0.0

        if n <= 2:
            for s in front:
                s.crowding_distance = float("inf")
            return

        # Sort by Latency
        front.sort(key=lambda s: s.latency_seconds)
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        lat_range = front[-1].latency_seconds - front[0].latency_seconds
        if lat_range > 0:
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    front[i + 1].latency_seconds - front[i - 1].latency_seconds
                ) / lat_range

        # Sort by Energy
        front.sort(key=lambda s: s.energy_joules)
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        energy_range = front[-1].energy_joules - front[0].energy_joules
        if energy_range > 0:
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    front[i + 1].energy_joules - front[i - 1].energy_joules
                ) / energy_range

    def select_optimal_tradeoff(self) -> FLConfigCandidate:
        """
        Select the best compromise configuration from the Pareto front using
        minimum normalized Euclidean distance to the ideal point (L_min, E_min).
        """
        if not self.pareto_front:
            self.fast_non_dominated_sort()

        self.calculate_crowding_distance(self.pareto_front)

        lats = [s.latency_seconds for s in self.pareto_front]
        energies = [s.energy_joules for s in self.pareto_front]

        min_l, max_l = min(lats), max(lats)
        min_e, max_e = min(energies), max(energies)

        range_l = max_l - min_l if max_l > min_l else 1.0
        range_e = max_e - min_e if max_e > min_e else 1.0

        best_score = float("inf")
        best_sol: Optional[FLConfigCandidate] = None

        for s in self.pareto_front:
            # Normalized Euclidean distance to ideal point (0, 0)
            norm_l = (s.latency_seconds - min_l) / range_l
            norm_e = (s.energy_joules - min_e) / range_e
            dist = (norm_l ** 2 + norm_e ** 2) ** 0.5

            if dist < best_score:
                best_score = dist
                best_sol = s

        # Mark selection
        for s in self.population:
            s.is_selected = False
        if best_sol is not None:
            best_sol.is_selected = True

        self.selected_solution = best_sol
        return best_sol

    def run_optimization(self) -> Tuple[List[FLConfigCandidate], FLConfigCandidate]:
        """Run complete EMO optimization pipeline and return (pareto_front, selected_solution)."""
        self.evaluate_candidates()
        self.fast_non_dominated_sort()
        selected = self.select_optimal_tradeoff()
        return self.pareto_front, selected

    def export_pareto_solutions(self, output_path: str | Path) -> None:
        """Export Pareto-optimal solutions to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "solution_id",
            "rounds",
            "local_epochs",
            "batch_size",
            "latency_seconds",
            "energy_joules",
            "energy_kj",
            "is_selected",
        ]

        # Sort pareto front by latency ascending
        sorted_front = sorted(self.pareto_front, key=lambda s: s.latency_seconds)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in sorted_front:
                writer.writerow({
                    "solution_id": s.solution_id,
                    "rounds": s.rounds,
                    "local_epochs": s.local_epochs,
                    "batch_size": s.batch_size,
                    "latency_seconds": s.latency_seconds,
                    "energy_joules": s.energy_joules,
                    "energy_kj": s.energy_kj,
                    "is_selected": s.is_selected,
                })
