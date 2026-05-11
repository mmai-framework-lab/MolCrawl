"""Per-task MCC bar chart for GUE reports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def plot_mcc_bar(task_metrics: Dict[str, Dict[str, float]], output_path: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    names = list(task_metrics)
    values = [task_metrics[name].get("mcc", task_metrics[name].get("f1_macro", float("nan"))) for name in names]
    fig, ax = plt.subplots(figsize=(max(4, 0.3 * len(names)), 3))
    ax.bar(names, values)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("MCC / macro F1")
    ax.set_title("GUE sub-task metrics")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
