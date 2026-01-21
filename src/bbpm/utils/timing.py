"""Timing utilities."""

import time
from contextlib import contextmanager
from typing import Generator, Optional


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.

        Args:
            name: Name/description of the operation being timed
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record elapsed time."""
        if self.start_time is not None:
            self.elapsed_time = time.perf_counter() - self.start_time
            print(f"{self.name} took {self.elapsed_time:.4f} seconds")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed_time is None:
            raise ValueError("Timer has not been used as context manager yet")
        return self.elapsed_time


@contextmanager
def timer(name: str = "Operation") -> Generator[Timer, None, None]:
    """
    Context manager for timing code blocks (convenience function).

    Args:
        name: Name/description of the operation being timed

    Yields:
        Timer instance
    """
    with Timer(name) as t:
        yield t
