"""Performance profiling utilities."""

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    import torch


@contextmanager
def timer(name: str, use_cuda_sync: bool = True) -> Generator[None, None, None]:
    """Context manager for profiling function execution time.

    Args:
        name: Name identifier for the profiled operation
        use_cuda_sync: If True, synchronize CUDA before timing (default: True)

    Yields:
        None

    Example:
        >>> with timer("my_function"):
        ...     result = expensive_operation()
    """
    import torch

    if use_cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    yield

    if use_cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}s")


def cuda_mem_gb() -> float:
    """Get current CUDA memory usage in GB.

    Returns:
        Memory usage in GB, or 0.0 if CUDA is not available
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


def tokens_per_second(num_tokens: int, elapsed_seconds: float) -> float:
    """Calculate tokens per second from token count and elapsed time.

    Args:
        num_tokens: Number of tokens processed
        elapsed_seconds: Elapsed time in seconds

    Returns:
        Tokens per second (0.0 if elapsed_seconds <= 0)
    """
    if elapsed_seconds <= 0:
        return 0.0
    return num_tokens / elapsed_seconds
