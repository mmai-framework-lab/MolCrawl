"""Placeholder visualisation for ChemLLMBench."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def plot_metric_bar(metrics: Dict[str, float], output_path: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not metrics:
        return None
    fig, ax = plt.subplots(figsize=(max(4, 0.3 * len(metrics)), 3))
    ax.bar(list(metrics.keys()), list(metrics.values()))
    ax.set_xticklabels(list(metrics.keys()), rotation=45, ha="right")
    ax.set_title("ChemLLMBench metrics")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
