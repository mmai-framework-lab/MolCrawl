"""Lightweight plots for the ClinVar evaluator.

The heavy-weight figure generation used by the legacy pipeline lives in
``molcrawl.evaluation.gpt2.clinvar_visualization``.  Here we only plot
what the new 3-axis report needs: the ROC curve and the score histogram.

Both functions silently become no-ops when ``matplotlib`` is not
available so the unit tests can run in a minimal environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_roc_curve(
    scores: np.ndarray, labels: np.ndarray, output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError:
        return None

    fpr, tpr, _thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUROC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ClinVar ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_score_histogram(
    scores: np.ndarray, labels: np.ndarray, output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(scores[labels == 1], bins=30, alpha=0.6, label="pathogenic")
    ax.hist(scores[labels == 0], bins=30, alpha=0.6, label="benign")
    ax.set_xlabel("pathogenicity score")
    ax.set_ylabel("count")
    ax.set_title("ClinVar pathogenicity score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
