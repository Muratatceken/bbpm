"""Abstract interfaces for memory operations.

This module defines the canonical interface for BBPM memory implementations.
Experiments should import IBbpmMemory and MemoryConfig from this module to
ensure compatibility across different memory implementations.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for BBPM memory.

    Attributes:
        num_blocks: Number of blocks (B)
        block_size: Size of each block (L), must be power of 2
        key_dim: Payload dimension (d)
        K: Number of slots per item
        H: Number of independent hash families
        dtype: Data type ("float32" or "bfloat16")
        device: Device ("cpu" or "cuda")
        normalize_values: Normalization mode ("none" | "l2" | "rms")
        read_mode: Read mode ("raw_mean" | "count_normalized")
        master_seed: Master seed (uint64)
        accumulate: Accumulation mode ("native" | "fast_inexact")
        output_dtype: Output dtype for read() ("float32" | "bfloat16")
    """

    num_blocks: int
    block_size: int
    key_dim: int  # d (payload dimension)
    K: int
    H: int
    dtype: str  # "float32" or "bfloat16"
    device: str  # "cpu" or "cuda"
    normalize_values: str  # "none" | "l2" | "rms"
    read_mode: str  # "raw_mean" | "count_normalized"
    master_seed: int
    accumulate: str = "native"  # "native" | "fast_inexact"
    output_dtype: str = "float32"  # "float32" | "bfloat16"

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not (self.block_size & (self.block_size - 1) == 0):
            raise ValueError(
                f"block_size must be power of 2, got {self.block_size}"
            )
        if self.key_dim <= 0:
            raise ValueError("key_dim must be positive")
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.H <= 0:
            raise ValueError("H must be positive")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError(
                f"dtype must be 'float32' or 'bfloat16', got {self.dtype}"
            )
        if self.device not in ("cpu", "cuda"):
            raise ValueError(
                f"device must be 'cpu' or 'cuda', got {self.device}"
            )
        # Validate CUDA availability if requested
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA device requested but CUDA is not available. "
                    "Use device='cpu' or ensure CUDA is properly installed."
                )
        if self.normalize_values not in ("none", "l2", "rms"):
            raise ValueError(
                f"normalize_values must be 'none', 'l2', or 'rms', got {self.normalize_values}"
            )
        if self.read_mode not in ("raw_mean", "count_normalized"):
            raise ValueError(
                f"read_mode must be 'raw_mean' or 'count_normalized', got {self.read_mode}"
            )
        if not (0 <= self.master_seed < 2**64):
            raise ValueError("master_seed must be uint64")
        if self.accumulate not in ("native", "fast_inexact"):
            raise ValueError(
                f"accumulate must be 'native' or 'fast_inexact', got {self.accumulate}"
            )
        if self.output_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                f"output_dtype must be 'float32' or 'bfloat16', got {self.output_dtype}"
            )
        # Validate bfloat16 requires fast_inexact
        if self.dtype == "bfloat16" and self.accumulate != "fast_inexact":
            raise ValueError(
                "bfloat16 requires accumulate='fast_inexact'. "
                "Safe accumulation with full-tensor casts is not supported "
                "(O(D*d) complexity violation)."
            )


class IBbpmMemory(Protocol):
    """Protocol for BBPM memory implementations.

    This protocol defines the canonical interface that all BBPM memory
    implementations must follow. Experiments should import and use this
    interface to ensure compatibility.

    Any class implementing this protocol must provide:
    - A `cfg` attribute of type MemoryConfig
    - `reset()` method to clear memory
    - `write(hx, v)` method to write values
    - `read(hx)` method to read values
    - `stats()` method to get statistics
    """

    cfg: MemoryConfig

    def reset(self) -> None:
        """Reset memory to initial state (clear all contents)."""
        ...

    def write(self, hx: int, v: "torch.Tensor") -> None:
        """Write value to memory at addresses derived from hashed key.

        Args:
            hx: Hashed item key (uint64)
            v: Value tensor of shape [d] (key_dim)
        """
        ...

    def read(self, hx: int) -> "torch.Tensor":
        """Read value from memory using hashed key.

        Args:
            hx: Hashed item key (uint64)

        Returns:
            Retrieved value tensor of shape [d] (key_dim)
        """
        ...

    def stats(self) -> dict:
        """Get memory statistics.

        Returns:
            Dictionary containing memory statistics (e.g., occupancy,
            collision rates, etc.)
        """
        ...
