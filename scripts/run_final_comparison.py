"""Run the frozen-data centralized, logical edge-cloud, and four-client FL study."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.final_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--central-epochs", type=int, default=20)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    args = parser.parse_args()
    rows, candidates = run_experiment(args.rounds, args.central_epochs,
                                      args.population, args.generations)
    print(f"Wrote results/final_comparison.csv ({len(rows)} rows)")
    print(f"Recorded {len(candidates)} measured candidate configurations")
    if not candidates or not any(c.get("energy_j") is not None for c in candidates):
        print("Energy unavailable: NSGA-II Pareto results are intentionally withheld.")


if __name__ == "__main__":
    main()
