"""Experiment 03: Runtime comparison vs attention."""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from bbpm.memory import BBPMemory, BBPMConfig


def run_exp03_runtime_vs_attention() -> Dict[str, float]:
    """Run runtime comparison experiment.

    Compares BBPM runtime to standard attention mechanisms.

    Returns:
        Dictionary containing experiment metrics
    """
    # TODO: Implement runtime comparison experiment
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    
    # Placeholder: create empty figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Exp 03: Runtime vs Attention")
    ax.grid(True)
    fig.savefig(output_dir / "figures" / "exp03_runtime_vs_attention.pdf")
    plt.close(fig)
    
    # Placeholder: create empty metrics
    metrics = {"sequence_lengths": [], "bbpm_times": [], "attention_times": []}
    with open(output_dir / "metrics" / "exp03_runtime_vs_attention.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    run_exp03_runtime_vs_attention()
