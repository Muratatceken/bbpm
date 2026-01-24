"""Device management utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def resolve_device(device: str) -> "torch.device":
    """Resolve device string to torch.device.

    Validates device string and returns corresponding torch.device.

    Args:
        device: Device string ("cpu" or "cuda")

    Returns:
        torch.device object

    Raises:
        ValueError: If device string is not "cpu" or "cuda"
        RuntimeError: If CUDA is requested but not available
    """
    import torch

    if device == "cpu":
        return torch.device("cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    else:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device}")


def resolve_dtype(dtype: str) -> "torch.dtype":
    """Resolve dtype string to torch.dtype.

    Validates dtype string and returns corresponding torch.dtype.

    Args:
        dtype: Data type string ("float32", "bfloat16", etc.)

    Returns:
        torch.dtype object

    Raises:
        ValueError: If dtype string is not recognized
    """
    import torch

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }

    if dtype not in dtype_map:
        raise ValueError(
            f"dtype must be one of {list(dtype_map.keys())}, got {dtype}"
        )

    return dtype_map[dtype]
