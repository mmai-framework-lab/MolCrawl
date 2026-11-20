"""Indel assay loading for ProteinGym (Issue #388, M3).

Indel DMS assay CSVs carry only ``mutated_sequence`` / ``DMS_score`` /
``DMS_score_bin`` -- no ``wildtype_sequence`` column and no ``mutant`` notation
to reconstruct one from. The reference (wildtype) comes from the release's DMS
reference file (``target_seq``) and is passed into ``load_proteingym`` as
``wildtype_sequence``. These tests pin that contract with no model or GPU.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from molcrawl.tasks.evaluation.proteingym.data_preparation import load_proteingym


def _write_indel_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "A4_HUMAN_Seuma_2022_indels.csv"
    pd.DataFrame(
        {
            # one deletion, one insertion -- both change length vs the wildtype
            "mutated_sequence": ["MLPGLALLL", "MLPGXLALLLL"],
            "DMS_score": [0.3, -1.2],
            "DMS_score_bin": [1, 0],
        }
    ).to_csv(csv, index=False)
    return csv


def test_indel_csv_uses_supplied_wildtype(tmp_path: Path):
    csv = _write_indel_csv(tmp_path)
    wt = "MLPGLALLLL"
    df = load_proteingym(csv, wildtype_sequence=wt)
    assert "wildtype_sequence" in df.columns
    assert (df["wildtype_sequence"] == wt).all()
    # DMS_score_bin is normalised to DMS_bin_score like the substitution path.
    assert "DMS_bin_score" in df.columns
    assert len(df) == 2


def test_indel_csv_without_wildtype_raises(tmp_path: Path):
    csv = _write_indel_csv(tmp_path)
    with pytest.raises(ValueError, match="no wildtype_sequence"):
        load_proteingym(csv)  # no wildtype supplied, no mutant column


def test_supplied_wildtype_ignored_when_column_present(tmp_path: Path):
    # A substitution-style CSV that already has wildtype_sequence must keep it;
    # a supplied override is only a fallback for the no-column case.
    csv = tmp_path / "subs.csv"
    pd.DataFrame(
        {
            "mutated_sequence": ["ACDE", "ACDF"],
            "wildtype_sequence": ["ACDF", "ACDF"],
            "DMS_score": [0.1, 0.2],
        }
    ).to_csv(csv, index=False)
    df = load_proteingym(csv, wildtype_sequence="ZZZZ")
    assert (df["wildtype_sequence"] == "ACDF").all()
