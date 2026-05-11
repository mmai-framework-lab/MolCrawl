"""Per-task bar chart helper for MoleculeNet reports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def plot_metric_bar(
    metrics: Dict[str, float], metric_name: str, output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    entries = {k: v for k, v in metrics.items() if k.endswith("." + metric_name) and not k.startswith("mean.")}
    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(max(4, 0.3 * len(entries)), 3))
    ax.bar(list(entries), list(entries.values()))
    ax.set_title(f"MoleculeNet - {metric_name}")
    ax.set_xticklabels(list(entries), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
