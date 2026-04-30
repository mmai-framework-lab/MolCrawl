"""Optional length-distribution plot for foldability reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def plot_length_distribution(
    sequences: Sequence[str], output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist([len(s) for s in sequences], bins=30)
    ax.set_xlabel("generated length")
    ax.set_ylabel("count")
    ax.set_title("Foldability length distribution")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
