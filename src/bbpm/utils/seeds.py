"""Seed management for determinism."""

import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def seed_everything(seed: int) -> None:
    """Set all random seeds for deterministic behavior.

    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).
    Also configures CuDNN for deterministic operations.

    Args:
        seed: Random seed value (should be non-negative integer)
    """
    import torch

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CuDNN deterministic mode
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
