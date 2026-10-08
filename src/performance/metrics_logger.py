"""
Metrics logger for performance and system benchmarking.
"""

import json
from pathlib import Path
from typing import Any, Dict


class MetricsLogger:
    """Structured metrics collector and serializer."""

    def __init__(self):
        self.records: Dict[str, Any] = {}

    def record(self, key: str, value: Any) -> None:
        self.records[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.records)

    def save_json(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=4)
