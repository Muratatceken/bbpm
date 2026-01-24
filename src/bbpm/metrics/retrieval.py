"""Retrieval quality metrics."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def cosine_similarity(a: "torch.Tensor", b: "torch.Tensor") -> float:
    """Compute cosine similarity between two tensors.

    Args:
        a: First tensor of shape [d] or [B, d]
        b: Second tensor of shape [d] or [B, d]

    Returns:
        Cosine similarity as float (in [-1, 1])
    """
    import torch

    # Cast to float32 for stable computation
    a = a.to(torch.float32)
    b = b.to(torch.float32)

    # Flatten if needed
    a_flat = a.flatten()
    b_flat = b.flatten()

    # Compute cosine similarity
    dot_product = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat, p=2)
    norm_b = torch.norm(b_flat, p=2)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return (dot_product / (norm_a * norm_b)).item()


def mse(a: "torch.Tensor", b: "torch.Tensor") -> float:
    """Compute mean squared error between two tensors.

    Args:
        a: First tensor of shape [d] or [B, d]
        b: Second tensor of shape [d] or [B, d]

    Returns:
        Mean squared error as float
    """
    import torch

    # Cast to float32 for stable computation
    a = a.to(torch.float32)
    b = b.to(torch.float32)

    return torch.mean((a - b) ** 2).item()


def snr_proxy(signal: "torch.Tensor", estimate: "torch.Tensor") -> float:
    """Compute signal-to-noise ratio proxy.

    SNR proxy = ||signal||^2 / ||signal - estimate||^2

    Args:
        signal: Ground truth signal tensor
        estimate: Estimated/reconstructed signal tensor

    Returns:
        SNR proxy as float (higher is better)
    """
    import torch

    # Cast to float32 for stable computation
    signal = signal.to(torch.float32)
    estimate = estimate.to(torch.float32)

    signal_power = torch.norm(signal, p=2) ** 2
    noise_power = torch.norm(signal - estimate, p=2) ** 2

    if noise_power == 0:
        return float("inf") if signal_power > 0 else 0.0

    return (signal_power / noise_power).item()


def summarize_trials(values: list[float]) -> dict:
    """Summarize multiple trial results with statistics.

    Args:
        values: List of float values from multiple trials

    Returns:
        Dictionary containing:
        - mean: Mean value
        - std: Standard deviation
        - ci95: 95% confidence interval (lower, upper)
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95": (0.0, 0.0)}

    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))  # Sample standard deviation

    # 95% confidence interval
    n = len(values)
    if n == 1:
        ci95 = (mean, mean)
    else:
        # Use t-distribution for CI95
        # For n >= 30, approximate with z-score (1.96)
        # For n < 30, use t-distribution approximation
        if n >= 30:
            z_critical = 1.96
            margin = z_critical * std / np.sqrt(n)
        else:
            # Approximate t-critical for small n
            # t(0.975, df) for common df values
            t_approx = {
                2: 4.303,
                3: 3.182,
                4: 2.776,
                5: 2.571,
                6: 2.447,
                7: 2.365,
                8: 2.306,
                9: 2.262,
                10: 2.228,
                15: 2.131,
                20: 2.086,
                25: 2.060,
                29: 2.045,
            }
            # Use closest approximation or default to 2.0
            t_critical = t_approx.get(n - 1, 2.0)
            margin = t_critical * std / np.sqrt(n)
        ci95 = (mean - margin, mean + margin)

    return {
        "mean": mean,
        "std": std,
        "ci95": tuple(ci95),
    }
