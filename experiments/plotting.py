"""Shared plotting utilities for experiments."""

import matplotlib
# Use Agg backend (non-interactive, PDF-compatible)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def get_git_commit() -> Optional[str]:
    """Get current git commit hash by walking up to find .git directory.
    
    Returns:
        Git commit hash string, or None if not found
    """
    current = Path(__file__).resolve()
    for _ in range(8):  # Max 8 levels up
        git_dir = current / ".git"
        if git_dir.exists():
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=current,
                )
                return result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None
        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent
    return None


def get_hardware_info() -> dict:
    """Get hardware and version information.
    
    Returns:
        Dictionary with torch_version, device, and optionally cuda_version, gpu_name
    """
    import torch
    info = {
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["device"] = "cuda"
    else:
        info["device"] = "cpu"
    return info


def save_pdf(fig, path: Path) -> None:
    """Save figure as PDF with tight layout.
    
    Args:
        fig: Matplotlib figure
        path: Output PDF path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def add_footer(fig, experiment_id: str, extra: Optional[dict] = None) -> None:
    """Add hardware/version footer to figure.
    
    Args:
        fig: Matplotlib figure
        experiment_id: Experiment identifier (e.g., "exp01")
        extra: Optional dictionary of additional info to include
    """
    hardware = get_hardware_info()
    git_commit = get_git_commit()
    
    footer_parts = [experiment_id]
    footer_parts.append(f"PyTorch {hardware['torch_version']}")
    footer_parts.append(f"Device: {hardware['device']}")
    
    if hardware["device"] == "cuda":
        footer_parts.append(f"CUDA {hardware['cuda_version']}")
        footer_parts.append(hardware['gpu_name'])
    
    if git_commit:
        footer_parts.append(f"Git: {git_commit[:8]}")
    
    if extra:
        for k, v in extra.items():
            footer_parts.append(f"{k}: {v}")
    
    footer_text = " | ".join(footer_parts)
    fig.text(0.5, 0.01, footer_text, ha="center", va="bottom", 
             fontsize=8, alpha=0.7)


def plot_line_with_ci(
    ax, x, mean, ci_low, ci_high, label: str, 
    linestyle: str = "-", color: Optional[str] = None
) -> None:
    """Plot line with confidence interval band.
    
    Args:
        ax: Matplotlib axes
        x: X values
        mean: Mean Y values
        ci_low: Lower CI bounds
        ci_high: Upper CI bounds
        label: Line label
        linestyle: Line style
        color: Optional color (if None, matplotlib chooses)
    """
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=2)
    ax.fill_between(x, ci_low, ci_high, alpha=0.2, color=color)
