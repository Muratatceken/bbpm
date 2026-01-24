"""Common utilities for experiments."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for bbpm imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from plotting import get_git_commit, get_hardware_info


def make_output_paths(out_dir: Path, exp_id: str, exp_slug: str) -> tuple[Path, Path]:
    """Create standardized output paths.
    
    Args:
        out_dir: Base output directory
        exp_id: Experiment ID (e.g., "exp01")
        exp_slug: Experiment slug (e.g., "snr_scaling")
        
    Returns:
        (metrics_path, figure_path)
    """
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = metrics_dir / f"{exp_id}_{exp_slug}.json"
    figure_path = figures_dir / f"{exp_id}_{exp_slug}.pdf"
    
    return metrics_path, figure_path


def seed_loop(num_seeds: int) -> List[int]:
    """Generate list of seeds for deterministic trials.
    
    Args:
        num_seeds: Number of seeds
        
    Returns:
        List of seed values [0, 1, ..., num_seeds-1]
    """
    return list(range(num_seeds))


def ensure_device(device_str: str) -> torch.device:
    """Get torch device, erroring if CUDA requested but unavailable.
    
    Args:
        device_str: "cpu" or "cuda"
        
    Returns:
        torch.device
        
    Raises:
        ValueError: If cuda requested but not available
    """
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but CUDA is not available. "
                "Use --device cpu or ensure CUDA is properly installed."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def dtype_from_string(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype.
    
    Args:
        dtype_str: "float32" or "bfloat16"
        
    Returns:
        torch.dtype
        
    Raises:
        ValueError: If dtype_str is unknown
    """
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def write_metrics_json(
    path: Path,
    experiment_id: str,
    config: Dict[str, Any],
    seeds: List[int],
    raw_trials: List[Dict[str, Any]],
    summary: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Write standardized metrics JSON.
    
    Args:
        path: Output JSON path
        experiment_id: Experiment identifier
        config: Experiment configuration
        seeds: List of seeds used
        raw_trials: List of per-trial results
        summary: Summary statistics with CI
        extra_info: Optional additional info to include
    """
    hardware = get_hardware_info()
    git_commit = get_git_commit()
    
    metrics = {
        "experiment_name": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "device": hardware["device"],
        "torch_version": hardware["torch_version"],
        "cuda_version": hardware.get("cuda_version"),
        "gpu_name": hardware.get("gpu_name"),
        "config": config,
        "seeds": seeds,
        "raw_trials": raw_trials,
        "summary": summary,
    }
    
    if extra_info:
        metrics["extra_info"] = extra_info
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
