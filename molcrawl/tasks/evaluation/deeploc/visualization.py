"""Confusion-matrix helper for DeepLoc reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return None
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("DeepLoc confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
