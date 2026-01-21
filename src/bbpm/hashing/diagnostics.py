"""Diagnostic functions for hash function analysis."""

from typing import Dict, List, Tuple

import numpy as np
import torch


def slot_loads(indices: torch.Tensor, D: int) -> np.ndarray:
    """
    Compute slot loads (counts per slot).

    Returns counts per slot. Only used for memory microbenchmarks,
    not for default experiment diagnostics.

    Args:
        indices: Flat tensor of indices [N]
        D: Total number of memory slots

    Returns:
        Array of counts per slot, shape [D]
    """
    indices_np = indices.cpu().numpy()
    hist, _ = np.histogram(indices_np, bins=D, range=(0, D))
    return hist


def occupancy_summary(
    indices: torch.Tensor, D: int, topk: int = 20, bins: int = 50
) -> Dict:
    """
    Compute compact occupancy summary (artifact-safe).

    Returns a compact dictionary with summary statistics instead of
    the full histogram array.

    Args:
        indices: Flat tensor of indices [N] or [B, K*H] (will be flattened)
        D: Total number of memory slots
        topk: Number of top-loaded slots to return
        bins: Number of bins for load histogram

    Returns:
        Dictionary with:
        - total_writes: int
        - unique_slots_touched: int
        - mean_load: float
        - std_load: float
        - max_load: int
        - top_slots: List[Tuple[int, int]] of (slot, load) pairs
        - load_hist_bins: List[float] (bin edges)
        - load_hist_counts: List[int] (counts per bin)
        - q2_estimate: float
        - collision_rate: float (within-row duplicates averaged)
    """
    indices_flat = indices.flatten()
    indices_np = indices_flat.cpu().numpy()
    total_writes = len(indices_np)

    if total_writes == 0:
        return {
            "total_writes": 0,
            "unique_slots_touched": 0,
            "mean_load": 0.0,
            "std_load": 0.0,
            "max_load": 0,
            "top_slots": [],
            "load_hist_bins": [],
            "load_hist_counts": [],
            "q2_estimate": 0.0,
            "collision_rate": 0.0,
        }

    # Compute histogram
    hist, _ = np.histogram(indices_np, bins=D, range=(0, D))
    unique_slots_touched = np.sum(hist > 0)

    # Basic stats
    mean_load = float(hist.mean())
    std_load = float(hist.std())
    max_load = int(hist.max())

    # Top K slots
    top_indices = np.argsort(hist)[-topk:][::-1]
    top_slots = [(int(slot), int(hist[slot])) for slot in top_indices if hist[slot] > 0]

    # Load histogram (distribution of loads)
    load_hist_counts, load_hist_bins = np.histogram(
        hist[hist > 0], bins=bins, range=(0, max_load + 1)
    )
    load_hist_bins = load_hist_bins.tolist()
    load_hist_counts = load_hist_counts.tolist()

    # Q2 estimate: computed from nonzero loads only
    # p_i = load_i / total_writes for nonzero loads, then q2 = sum(p_i^2)
    nonzero_loads = hist[hist > 0]
    if len(nonzero_loads) > 0:
        probs = nonzero_loads / total_writes
        q2_estimate = float(np.sum(probs ** 2))
    else:
        q2_estimate = 0.0

    # Collision rate (within-row duplicates)
    if indices.dim() == 2:
        # For [B, K*H] shape, compute collision rate per row and average
        collision_rates = []
        for row in indices:
            flat_row = row.flatten()
            unique_count = torch.unique(flat_row).numel()
            total_count = flat_row.numel()
            if total_count > 0:
                collision_rates.append(1.0 - (unique_count / total_count))
        collision_rate_val = float(np.mean(collision_rates)) if collision_rates else 0.0
    else:
        # For flat tensor, use global collision rate
        unique_count = torch.unique(indices_flat).numel()
        collision_rate_val = 1.0 - (unique_count / total_writes)

    return {
        "total_writes": int(total_writes),
        "unique_slots_touched": int(unique_slots_touched),
        "mean_load": mean_load,
        "std_load": std_load,
        "max_load": max_load,
        "top_slots": top_slots,
        "load_hist_bins": load_hist_bins,
        "load_hist_counts": load_hist_counts,
        "q2_estimate": q2_estimate,
        "collision_rate": collision_rate_val,
    }


def gini_of_load(loads: np.ndarray) -> float:
    """
    Compute Gini coefficient from load array.

    Args:
        loads: Array of load values

    Returns:
        Gini coefficient (0 to 1)
    """
    loads = loads[loads > 0]  # Only non-zero loads

    if len(loads) == 0:
        return 0.0

    sorted_loads = np.sort(loads)
    n = len(sorted_loads)
    cumsum = np.cumsum(sorted_loads)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_loads)) / (n * cumsum[-1]) - (
        n + 1
    ) / n

    return float(gini)


def self_collision_prob(K: int, L: int) -> float:
    """
    Estimate self-collision probability: ~1 - exp(-K(K-1)/(2L)).

    This estimates the probability that K items hashing to L slots
    will have at least one collision among themselves.

    Args:
        K: Number of items/slots per key
        L: Number of available slots (e.g., block_size or D)

    Returns:
        Self-collision probability (0 to 1)
    """
    if L <= 0 or K <= 0:
        return 0.0
    if K == 1:
        return 0.0
    # Approximation: 1 - exp(-K(K-1)/(2L))
    exponent = -K * (K - 1) / (2.0 * L)
    return float(1.0 - np.exp(exponent))


def occupancy_hist(indices: torch.Tensor, D: int) -> Dict[str, int]:
    """
    DEPRECATED: Use occupancy_summary() instead for artifact-safe diagnostics.

    Compute occupancy histogram.

    Counts how many times each slot is addressed.

    Args:
        indices: Flat tensor of indices [N]
        D: Total number of memory slots

    Returns:
        Dictionary with histogram statistics
    """
    indices_np = indices.cpu().numpy()
    hist, _ = np.histogram(indices_np, bins=D, range=(0, D))

    return {
        "histogram": hist.tolist(),
        "mean": float(hist.mean()),
        "std": float(hist.std()),
        "min": int(hist.min()),
        "max": int(hist.max()),
    }


def max_load(indices: torch.Tensor, D: int) -> int:
    """
    Compute maximum load (number of items hashing to the most loaded slot).

    Args:
        indices: Flat tensor of indices [N]
        D: Total number of memory slots

    Returns:
        Maximum load value
    """
    indices_np = indices.cpu().numpy()
    hist, _ = np.histogram(indices_np, bins=D, range=(0, D))
    return int(hist.max())


def gini_load(indices: torch.Tensor, D: int) -> float:
    """
    Compute Gini coefficient of load distribution.

    Measures inequality in slot occupancy (0 = uniform, 1 = maximum inequality).

    Args:
        indices: Flat tensor of indices [N]
        D: Total number of memory slots

    Returns:
        Gini coefficient (0 to 1)
    """
    indices_np = indices.cpu().numpy()
    hist, _ = np.histogram(indices_np, bins=D, range=(0, D))
    hist = hist[hist > 0]  # Only non-zero loads

    if len(hist) == 0:
        return 0.0

    # Sort for Gini calculation
    sorted_hist = np.sort(hist)
    n = len(sorted_hist)
    cumsum = np.cumsum(sorted_hist)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_hist)) / (n * cumsum[-1]) - (n + 1) / n

    return float(gini)


def collision_rate(indices: torch.Tensor) -> float:
    """
    Compute collision rate within a batch.

    Measures fraction of duplicate indices within the batch.

    Args:
        indices: Indices tensor of shape [B, K*H]

    Returns:
        Collision rate (0 to 1)
    """
    # Flatten and count unique vs total
    flat = indices.flatten()
    unique_count = torch.unique(flat).numel()
    total_count = flat.numel()

    if total_count == 0:
        return 0.0

    return 1.0 - (unique_count / total_count)


def estimate_q2(indices: torch.Tensor, D: int) -> float:
    """
    Estimate sum of squared slot probabilities (proxy for collision probability).

    Computes sum_i (c_i / total)^2 where c_i is count for slot i.
    This is a proxy for ∑ q_i^2 used in theoretical analysis.

    Args:
        indices: Flat tensor of indices [N]
        D: Total number of memory slots

    Returns:
        Estimated q^2 value
    """
    indices_np = indices.cpu().numpy()
    hist, _ = np.histogram(indices_np, bins=D, range=(0, D))
    total = len(indices_np)

    if total == 0:
        return 0.0

    # Normalize to probabilities
    probs = hist / total
    # Sum of squares
    q2 = np.sum(probs ** 2)

    return float(q2)
