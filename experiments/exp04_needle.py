"""Experiment 04: Needle-in-haystack retrieval."""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from bbpm.memory import BBPMemory, BBPMConfig


def run_exp04_needle() -> Dict[str, float]:
    """Run needle-in-haystack experiment.

    Tests retrieval of a specific item from a large haystack of noise.

    Returns:
        Dictionary containing experiment metrics
    """
    # TODO: Implement needle-in-haystack experiment
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    
    # Placeholder: create empty figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Haystack Size")
    ax.set_ylabel("Retrieval Accuracy")
    ax.set_title("Exp 04: Needle-in-Haystack")
    ax.grid(True)
    fig.savefig(output_dir / "figures" / "exp04_needle.pdf")
    plt.close(fig)
    
    # Placeholder: create empty metrics
    metrics = {"haystack_sizes": [], "accuracies": []}
    with open(output_dir / "metrics" / "exp04_needle.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    run_exp04_needle()
