"""Scatter of predicted vs observed delta for Replogle reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def plot_delta_scatter(
    observed: np.ndarray, predicted: np.ndarray, output_path: Path,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(observed.flatten(), predicted.flatten(), s=2, alpha=0.3)
    ax.set_xlabel("observed delta")
    ax.set_ylabel("predicted delta")
    ax.set_title("Replogle Perturb-seq delta")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _noqa(_: Sequence[float]):  # pragma: no cover - keep import surface stable
    pass
