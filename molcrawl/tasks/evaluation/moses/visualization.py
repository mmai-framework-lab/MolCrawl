"""Optional distribution-summary plots for MOSES."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def plot_length_histogram(
    smiles: Sequence[str], output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    lengths = [len(s) for s in smiles]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(lengths, bins=40)
    ax.set_xlabel("SMILES length (chars)")
    ax.set_ylabel("count")
    ax.set_title("Generated SMILES length distribution")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
