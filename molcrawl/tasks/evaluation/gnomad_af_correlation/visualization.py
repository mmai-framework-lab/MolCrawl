"""Scatter plot helper for gnomAD AF correlation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def plot_scatter(
    af: Sequence[float], scores: Sequence[float], output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(list(af), list(scores), s=5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("allele frequency (log)")
    ax.set_ylabel("model score (var - ref LL)")
    ax.set_title("gnomAD allele frequency correlation")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
