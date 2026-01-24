"""Memory occupancy metrics."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def gini(counts: list[int]) -> float:
    """Compute Gini coefficient for inequality measure.

    Gini coefficient measures inequality in distribution.
    Range: [0, 1] where 0 = perfect equality, 1 = maximum inequality.

    Args:
        counts: List of non-negative integer counts

    Returns:
        Gini coefficient as float
    """
    if not counts:
        return 0.0

    # Sort counts in ascending order
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    total = sum(sorted_counts)

    if total == 0:
        return 0.0

    # Compute Gini coefficient using formula:
    # G = (2 * sum((i+1) * x_i)) / (n * sum(x_i)) - (n + 1) / n
    numerator = sum((i + 1) * x for i, x in enumerate(sorted_counts))
    gini = (2 * numerator) / (n * total) - (n + 1) / n

    return float(gini)


def overlap_rate(a: list[int], b: list[int]) -> float:
    """Compute overlap rate between two sets (Jaccard similarity).

    overlap_rate = |A ∩ B| / |A ∪ B|

    Args:
        a: First list of integers (treated as set)
        b: Second list of integers (treated as set)

    Returns:
        Overlap rate (Jaccard similarity) in [0, 1]
    """
    set_a = set(a)
    set_b = set(b)

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union


def block_occupancy(block_ids: list[int], num_blocks: int) -> dict:
    """Analyze block occupancy distribution.

    Args:
        block_ids: List of block IDs (may contain duplicates)
        num_blocks: Total number of blocks

    Returns:
        Dictionary containing:
        - counts_per_block: List of counts for each block [0, num_blocks)
        - max_occupancy: Maximum count in any block
        - mean_occupancy: Mean count per block
        - gini_coefficient: Gini coefficient of occupancy distribution
    """
    # Count occurrences of each block ID
    counts = [0] * num_blocks
    for block_id in block_ids:
        if 0 <= block_id < num_blocks:
            counts[block_id] += 1

    max_occupancy = max(counts) if counts else 0
    mean_occupancy = sum(counts) / num_blocks if num_blocks > 0 else 0.0
    gini_coeff = gini(counts)

    return {
        "counts_per_block": counts,
        "max_occupancy": max_occupancy,
        "mean_occupancy": mean_occupancy,
        "gini_coefficient": gini_coeff,
    }
