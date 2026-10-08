"""
High-precision benchmarking timer.
"""

from contextlib import contextmanager
import time
from typing import Optional


class BenchmarkTimer:
    """Timer for measuring execution duration with perf_counter."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_seconds: float = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        self.end_time = time.perf_counter()
        self.duration_seconds = self.end_time - self.start_time
        return self.duration_seconds

    @property
    def elapsed_ms(self) -> float:
        return self.duration_seconds * 1000.0


@contextmanager
def measure_time():
    """Context manager yielding a dictionary with 'duration_seconds' in seconds."""
    result = {"duration_seconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["duration_seconds"] = time.perf_counter() - start
