"""Small monotonic elapsed-time utilities."""

from __future__ import annotations

import time


class Timer:
    """Measure elapsed time with ``perf_counter``.

    A timer can be started/stopped directly or used as a context manager. Each
    ``start`` resets the timer; ``stop`` returns milliseconds.
    """

    def __init__(self, *, start: bool = False) -> None:
        self._started_at: float | None = None
        self.elapsed_ms: float | None = None
        if start:
            self.start()

    def start(self) -> "Timer":
        self._started_at = time.perf_counter()
        self.elapsed_ms = None
        return self

    def stop(self) -> float:
        if self._started_at is None:
            raise RuntimeError("Timer has not been started")
        self.elapsed_ms = (time.perf_counter() - self._started_at) * 1000.0
        self._started_at = None
        return self.elapsed_ms

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def elapsed_ms(started_at: float) -> float:
    """Return milliseconds elapsed since a ``time.perf_counter()`` reading."""
    return (time.perf_counter() - started_at) * 1000.0
