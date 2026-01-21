"""Device detection and management utilities."""

import torch
from typing import Literal

Device = Literal["cpu", "cuda"]


def get_device(device: str = "auto") -> str:
    """
    Get the appropriate device string.

    Args:
        device: Device string ("cpu", "cuda", or "auto"). If "auto",
                selects CUDA if available, otherwise CPU.

    Returns:
        Device string ("cpu" or "cuda")
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return device
    else:
        raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda', or 'auto'")


def set_device(device: str) -> torch.device:
    """
    Get a torch.device object.

    Args:
        device: Device string ("cpu", "cuda", or "auto")

    Returns:
        torch.device object
    """
    device_str = get_device(device)
    return torch.device(device_str)
