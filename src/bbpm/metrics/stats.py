"""Statistical utilities for experiments."""

import numpy as np
from typing import Sequence, Dict, Any, Tuple
from collections import defaultdict


def gini_coefficient(values: Sequence[float]) -> float:
    """Compute Gini coefficient for inequality measurement.
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Gini coefficient in [0, 1] (0 = perfect equality, 1 = maximum inequality)
    """
    values = np.array(values)
    if len(values) == 0 or np.all(values == 0):
        return 0.0
    
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n


def mean_ci95(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """Compute mean and 95% confidence interval using normal approximation.
    
    Uses z=1.96 for all sample sizes (acceptable for n>=10 in ML papers).
    This is a normal approximation; for small samples (n<30), t-distribution would
    be more accurate, but normal approximation is standard in ML practice.
    
    Args:
        values: Array of float values
        
    Returns:
        (mean, ci_low, ci_high, std)
    """
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    
    if n == 1:
        return (mean, mean, mean, 0.0)
    
    std = float(np.std(arr, ddof=1))  # Sample standard deviation
    se = std / np.sqrt(n)  # Standard error
    z = 1.96  # Normal approximation for 95% CI
    margin = z * se
    
    return (mean, mean - margin, mean + margin, std)


def summarize_groups(
    raw_rows: list[Dict[str, Any]],
    groupby_keys: list[str],
    metric_keys: list[str]
) -> Dict[Tuple, Dict[str, Dict[str, float]]]:
    """Summarize metrics grouped by specified keys.
    
    Groups raw trial data by specified keys and computes mean/CI/std for each
    metric within each group.
    
    Args:
        raw_rows: List of trial dictionaries
        groupby_keys: Keys to group by (e.g., ["N", "K"])
        metric_keys: Metric keys to summarize (e.g., ["cosine", "mse"])
        
    Returns:
        Dictionary keyed by group tuple (e.g., (N=1000, K=32))
        with nested dict: {metric_name: {mean, ci95_low, ci95_high, std}}
    """
    # Group rows by groupby_keys
    groups = defaultdict(list)
    for row in raw_rows:
        group_key = tuple(row[k] for k in groupby_keys)
        groups[group_key].append(row)
    
    # Summarize each group
    result = {}
    for group_key, group_rows in groups.items():
        summaries = {}
        for metric_key in metric_keys:
            values = [row[metric_key] for row in group_rows if metric_key in row]
            if values:
                mean, ci_low, ci_high, std = mean_ci95(values)
                summaries[metric_key] = {
                    "mean": mean,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "std": std,
                }
        result[group_key] = summaries
    
    return result
