"""Metrics module for BBPM."""

from bbpm.metrics.occupancy import block_occupancy, gini, overlap_rate
from bbpm.metrics.retrieval import cosine_similarity, mse, snr_proxy, summarize_trials
from bbpm.metrics.stats import gini_coefficient, mean_ci95, summarize_groups

__all__ = [
    "cosine_similarity",
    "mse",
    "snr_proxy",
    "summarize_trials",
    "gini",
    "overlap_rate",
    "block_occupancy",
    "gini_coefficient",
    "mean_ci95",
    "summarize_groups",
]
