"""Per-group perplexity bar chart."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def plot_perplexity_bar(metrics: Dict[str, float], output_path: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    entries = {
        k[: -len(".perplexity")]: v for k, v in metrics.items() if k.endswith(".perplexity") and not k.startswith("mean.")
    }
    if not entries:
        return None
    fig, ax = plt.subplots(figsize=(max(4, 0.3 * len(entries)), 3))
    ax.bar(list(entries), list(entries.values()))
    ax.set_xticklabels(list(entries), rotation=45, ha="right")
    ax.set_ylabel("perplexity")
    ax.set_title("rna_benchmark per-group perplexity")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
