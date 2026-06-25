"""Unit tests for the migrated ClinVar task.

The full evaluator requires a trained GPT-2 checkpoint, so here we test
the pieces that do not need torch: the data loader, the splits helper,
and the threshold / confusion-matrix utilities.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from molcrawl.tasks.evaluation.clinvar.predictions_log import _write_jsonl
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


def test_load_clinvar_preserves_metadata_columns(tmp_path):
    """vcv_id / review_status / consequence must survive the load step."""
    df_in = _sample_df().assign(
        vcv_id=[
            "VCV000000001",
            "VCV000000002",
            "VCV000000003",
            "VCV000000004",
        ],
        review_status=[
            "criteria_provided,_single_submitter",
            "reviewed_by_expert_panel",
            "no_assertion_criteria_provided",
            "criteria_provided,_multiple_submitters,_no_conflicts",
        ],
        consequence=[
            "missense_variant",
            "synonymous_variant",
            "intron_variant",
            "missense_variant",
        ],
    )
    src = tmp_path / "clinvar_with_metadata.csv"
    df_in.to_csv(src, index=False)
    df = load_clinvar(str(src))
    # 3 rows survive (Uncertain significance dropped), but every metadata
    # column is still on the frame for downstream traceability / joins.
    assert len(df) == 3
    for col in ("vcv_id", "review_status", "consequence"):
        assert col in df.columns, f"{col} must survive load_clinvar"
    # vcv_id and the canonical NCBI prefix shape are intact.
    assert df["vcv_id"].iloc[0].startswith("VCV")
    # The pathogenic label join still works.
    assert set(df["pathogenic"]) <= {0, 1}


def test_load_clinvar_tolerates_legacy_csv_without_metadata(tmp_path):
    """Old CSVs (pre-vcv_id) still load — metadata cols are just absent."""
    src = tmp_path / "legacy.csv"
    _sample_df().to_csv(src, index=False)
    df = load_clinvar(str(src))
    assert "vcv_id" not in df.columns
    assert "reference_sequence" in df.columns  # sequences still load


def test_predictions_jsonl_includes_vcv_metadata(tmp_path):
    """Every per-variant JSONL record carries vcv_id / review_status / consequence."""
    df = pd.DataFrame(
        {
            "vcv_id": ["VCV000000001", "VCV000000002"],
            "review_status": [
                "criteria_provided,_single_submitter",
                "reviewed_by_expert_panel",
            ],
            "consequence": ["missense_variant", "synonymous_variant"],
            "chrom": ["1", "2"],
            "pos": [100, 200],
            "ref": ["A", "G"],
            "alt": ["T", "C"],
            "reference_sequence": ["AAA", "GGG"],
            "variant_sequence": ["ATA", "GCG"],
            "ClinicalSignificance": ["Pathogenic", "Benign"],
        }
    )
    ref_ll = np.array([-1.0, -1.5])
    var_ll = np.array([-2.0, -1.4])
    scores = ref_ll - var_ll
    labels = np.array([1, 0])
    predicted = np.array([1, 0])
    correct = np.array([True, True])

    out = tmp_path / "predictions.jsonl"
    _write_jsonl(
        out, df, ref_ll, var_ll, scores, labels,
        threshold=0.0, predicted=predicted, correct=correct,
    )
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    for rec, expected_vcv in zip(records, ["VCV000000001", "VCV000000002"]):
        assert rec["vcv_id"] == expected_vcv
        assert rec["review_status"] is not None
        assert rec["consequence"] is not None


def test_predictions_jsonl_handles_legacy_rows_without_vcv(tmp_path):
    """Legacy DataFrames without vcv_id still serialise — fields are null."""
    df = pd.DataFrame(
        {
            "chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["T"],
            "reference_sequence": ["AAA"], "variant_sequence": ["ATA"],
            "ClinicalSignificance": ["Pathogenic"],
        }
    )
    out = tmp_path / "predictions.jsonl"
    _write_jsonl(
        out, df,
        ref_ll=np.array([-1.0]), var_ll=np.array([-2.0]),
        scores=np.array([1.0]), labels=np.array([1]),
        threshold=None, predicted=None, correct=None,
    )
    rec = json.loads(out.read_text().strip())
    assert rec["vcv_id"] is None
    assert rec["chrom"] == "1"  # sequence-side metadata still serialised


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
