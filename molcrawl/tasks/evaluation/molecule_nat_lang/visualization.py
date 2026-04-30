"""Optional histogram of molecule-caption log-likelihoods."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def plot_likelihood_histogram(
    log_likelihoods: Sequence[float], output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(list(log_likelihoods), bins=30)
    ax.set_xlabel("mean per-token log-likelihood")
    ax.set_ylabel("count")
    ax.set_title("molecule_nat_lang pair scores")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
