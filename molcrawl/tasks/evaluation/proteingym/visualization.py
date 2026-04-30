"""ProteinGym visualisation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def plot_score_vs_dms(
    scores: Sequence[float], dms: Sequence[float], output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(list(dms), list(scores), s=5, alpha=0.6)
    ax.set_xlabel("DMS score")
    ax.set_ylabel("model score (log-likelihood diff)")
    ax.set_title("ProteinGym correlation")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
