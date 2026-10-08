"""
Evaluation, architecture comparisons, and report generation module.
"""

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

__all__ = [
    "validate_architecture_ordering",
    "save_final_architecture_csvs",
    "plot_final_graphs",
    "FRAMEWORK_LABELS",
    "generate_final_experiment_config",
    "generate_report_ready_results",
]
