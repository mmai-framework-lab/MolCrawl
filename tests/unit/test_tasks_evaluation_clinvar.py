"""Unit tests for the migrated ClinVar task.

The full evaluator requires a trained GPT-2 checkpoint, so here we test
the pieces that do not need torch: the data loader, the splits helper,
and the threshold / confusion-matrix utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molcrawl.tasks.evaluation.clinvar.data_preparation import (
    add_pathogenic_label,
    load_clinvar,
)
from molcrawl.tasks.evaluation.clinvar.metrics import (
    confusion_summary,
    find_optimal_f1_threshold,
    sensitivity_specificity,
)
from molcrawl.tasks.evaluation.clinvar.splits import chromosome_split


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_sequence": ["ACGT", "ACGT", "ACGT", "ACGT"],
            "variant_sequence": ["ACAT", "ACCT", "ACGA", "ACGC"],
            "ClinicalSignificance": [
                "Pathogenic",
                "Likely benign",
                "Uncertain significance",
                "Benign",
            ],
            "Chromosome": ["chr1", "chr21", "chrX", "chr2"],
        }
    )


def test_add_pathogenic_label_drops_unknown():
    df = add_pathogenic_label(_sample_df())
    assert set(df["pathogenic"].unique()) <= {0, 1}
    assert "Uncertain significance" not in df["ClinicalSignificance"].values


def test_load_clinvar_csv(tmp_path):
    src = tmp_path / "clinvar.csv"
    _sample_df().to_csv(src, index=False)
    df = load_clinvar(str(src))
    assert len(df) == 3  # Uncertain significance dropped


def test_load_clinvar_missing_columns(tmp_path):
    src = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(src, index=False)
    with pytest.raises(ValueError):
        load_clinvar(str(src))


def test_chromosome_split_partitions_unseen():
    df = add_pathogenic_label(_sample_df())
    seen, unseen = chromosome_split(df, unseen=("chr21", "chrX"))
    assert {"chr1", "chr2"} >= set(seen["Chromosome"])
    assert set(unseen["Chromosome"]) <= {"chr21", "chrX"}


def test_find_optimal_f1_threshold_recovers_perfect_split():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    threshold = find_optimal_f1_threshold(scores, labels)
    preds = (scores > threshold).astype(int)
    assert (preds == labels).all()


def test_confusion_summary_keys():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 0])
    cm = confusion_summary(labels, preds)
    assert set(cm) == {"true_negative", "false_positive", "false_negative", "true_positive"}
    sensitivity, specificity = sensitivity_specificity(labels, preds)
    assert 0.0 <= sensitivity <= 1.0
    assert 0.0 <= specificity <= 1.0
