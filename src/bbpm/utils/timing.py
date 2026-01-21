"""Timing utilities."""

import time
from contextlib import contextmanager
from typing import Generator, Optional

import torch


class Timer:
    """Context manager for timing code blocks with CUDA synchronization support."""

    def __init__(self, name: str = "Operation", device: str = "cpu"):
        """
        Initialize timer.

        Args:
            name: Name/description of the operation being timed
            device: Device string ("cpu", "cuda", or "auto"). If "cuda" or "auto" with CUDA available,
                    synchronizes GPU operations for accurate timing.
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
        
        # Auto-detect device if "auto"
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def __enter__(self) -> "Timer":
        """Start timing."""
        # Synchronize CUDA before starting timer
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record elapsed time."""
        if self.start_time is not None:
            # Synchronize CUDA before stopping timer
            if self.device == "cuda":
                torch.cuda.synchronize()
            self.elapsed_time = time.perf_counter() - self.start_time
            print(f"{self.name} took {self.elapsed_time:.4f} seconds")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed_time is None:
            raise ValueError("Timer has not been used as context manager yet")
        return self.elapsed_time


@contextmanager
def timer(name: str = "Operation", device: str = "cpu") -> Generator[Timer, None, None]:
    """
    Context manager for timing code blocks (convenience function).

    Args:
        name: Name/description of the operation being timed
        device: Device string ("cpu", "cuda", or "auto")

    Yields:
        Timer instance
    """
    with Timer(name, device=device) as t:
        yield t
