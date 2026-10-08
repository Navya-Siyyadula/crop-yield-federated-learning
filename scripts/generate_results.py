"""
Result generation CLI script.
Runs or regenerates the final three-architecture results and figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import run_experiment


def main():
    run_experiment()


if __name__ == "__main__":
    main()
