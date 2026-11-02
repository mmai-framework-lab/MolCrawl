"""Finding the tables that put two architectures side by side.

The first version of this inventory was hand-picked and missed a dozen tables, so
what these cover is the rule that replaced the hand-picking: when the architecture
counts as an axis of the table, and when it does not.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    path = repo_root() / "scripts" / "arch_side_by_side_scan.py"
    spec = importlib.util.spec_from_file_location("arch_side_by_side_scan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(mod, tmp_path: Path, files: dict) -> list[dict]:
    root = tmp_path / "report"
    root.mkdir(exist_ok=True)
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    out = tmp_path / "out.tsv"
    assert mod.main(["--root", str(root), "--out", str(out)]) == 0
    return list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines(),
                               delimiter="\t"))


# --- when the architecture is an axis ----------------------------------------

def test_a_header_naming_both_architectures_is_an_axis(mod, tmp_path):
    rows = run(mod, tmp_path, {"a.md": """
| | BERT | GPT-2 |
|---|---|---|
| compounds | 0.0674 | 0.5507 |
"""})
    assert len(rows) == 1
    assert rows[0]["axis"] == "列（見出しが arch）"
    assert rows[0]["modality_scope"] == "この表の中"


def test_an_arch_column_with_both_values_is_an_axis(mod, tmp_path):
    rows = run(mod, tmp_path, {"a.md": """
| arch | size | 採用値 |
|---|---|---|
| bert | small | 0.067367 |
| gpt2 | small | 0.550657 |
| | | compounds |
"""})
    assert len(rows) == 1
    assert rows[0]["axis"] == "列（arch）"
    assert rows[0]["subject"] == "事前学習 loss"


def test_rows_keyed_by_architecture_are_an_axis(mod, tmp_path):
    rows = run(mod, tmp_path, {"a.md": """
| | 採用値 |
|---|---|
| BERT compounds | 0.067367 |
| GPT-2 compounds | 0.550657 |
"""})
    assert len(rows) == 1
    assert rows[0]["axis"] == "行（先頭セルが arch）"


def test_one_cell_holding_both_is_recorded_as_not_an_axis(mod, tmp_path):
    """corpus-size's epochs cell: the rows are modalities, not architectures."""
    rows = run(mod, tmp_path, {
        "corpus.tsv": "modality\tepochs\n"
                      "compounds\tBERT: 288.78 ｜ GPT-2: 29.99\n"})
    assert len(rows) == 1
    assert rows[0]["axis"] == mod.IN_ONE_CELL
    assert "1 セルに両方" in rows[0]["rows"]


# --- when it is not ----------------------------------------------------------

def test_one_architecture_alone_is_not_a_comparison(mod, tmp_path):
    assert run(mod, tmp_path, {"a.md": """
| arch | 採用値 |
|---|---|
| bert | 0.067367 |
| | compounds |
"""}) == []


def test_a_file_listing_is_not_an_axis(mod, tmp_path):
    """The names appear inside prose cells and no column belongs to either."""
    assert run(mod, tmp_path, {"a.md": """
| ファイル | 内容 |
|---|---|
| curves.tsv | compounds の BERT と GPT-2 の曲線 |
| ladder.tsv | 両方の採用値 |
"""}) == []


def test_a_prose_cell_naming_both_without_values_is_not_a_comparison(mod, tmp_path):
    """The single-cell rule needs a number after each name, or every sentence that
    mentions the two architectures counts as putting them side by side."""
    assert run(mod, tmp_path, {"a.md": """
| 項 | 内容 |
|---|---|
| 方針 | compounds の BERT と GPT-2 は同じ軸に描かない |
"""}) == []


def test_a_cell_with_a_value_against_each_name_is_a_comparison(mod, tmp_path):
    rows = run(mod, tmp_path, {"a.md": """
| modality | epochs |
|---|---|
| compounds | BERT: 288.78（45,000 step）｜ GPT-2: 29.99（4,674 iter） |
"""})
    assert len(rows) == 1
    assert rows[0]["axis"] == mod.IN_ONE_CELL


def test_a_table_without_the_modality_anywhere_is_left_out(mod, tmp_path):
    assert run(mod, tmp_path, {"other.md": """
| | BERT | GPT-2 |
|---|---|---|
| rna | 2.6957 | 3.1398 |
"""}) == []


def test_a_table_in_a_modality_named_file_counts_with_the_scope_said(mod, tmp_path):
    rows = run(mod, tmp_path, {"compounds-notes.md": """
| | BERT | GPT-2 |
|---|---|---|
| step | 45,000 | 4,674 |
"""})
    assert len(rows) == 1
    assert rows[0]["modality_scope"] == "この文書全体"


def test_a_tsv_with_no_header_row_is_not_read_as_an_arch_column(mod, tmp_path):
    """submissions-*.tsv has no header; a job name holding "bert" is not a column."""
    assert run(mod, tmp_path, {
        "submissions.tsv": "266543\tds45k-solubility-bert-small\t13:23\n"
                           "266551\tds45k-solubility-gpt2-small\t13:23\n"}) == []


def test_the_output_table_does_not_list_itself(mod, tmp_path):
    root = tmp_path / "report"
    root.mkdir()
    (root / "a.md").write_text("""
| arch | 採用値 |
|---|---|
| bert | 0.067367 |
| gpt2 | 0.550657 |
| | compounds |
""", encoding="utf-8")
    out = root / "side-by-side.tsv"
    assert mod.main(["--root", str(root), "--out", str(out)]) == 0
    assert mod.main(["--root", str(root), "--out", str(out)]) == 0   # second pass
    rows = list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines(),
                               delimiter="\t"))
    assert [r["file"] for r in rows] == ["a.md"]


# --- the subject -------------------------------------------------------------

def test_a_key_column_is_not_the_subject(mod, tmp_path):
    """An "lr" column says which row, not that the table is about learning rates."""
    rows = run(mod, tmp_path, {
        "compounds-loss-curves.tsv":
            "arch\tsize\tlr\tseed\tstep\tsplit\tvalue\tsource\n"
            "bert\tsmall\t1e-3\t42\t100\tvalid\t0.0674\tcompounds/checkpoint-100\n"
            "gpt2\tsmall\t8e-4\t42\t100\tvalid\t0.5507\tcompounds/ckpt.pt\n"})
    assert len(rows) == 1
    assert rows[0]["subject"] == "事前学習 loss"


def test_the_subject_comes_from_the_cells_when_the_header_is_terse(mod, tmp_path):
    rows = run(mod, tmp_path, {
        "x.tsv": "task\tmetric\tarch\ttrained\n"
                 "herg\tROC-AUC\tbert\t0.8705\n"
                 "herg\tROC-AUC\tgpt2\t0.8528\n"})
    assert len(rows) == 1
    assert rows[0]["subject"] == "下流"


def test_an_axis_with_no_keyword_is_said_to_be_unclassified(mod, tmp_path):
    rows = run(mod, tmp_path, {"a.md": """
| | BERT | GPT-2 |
|---|---|---|
| compounds | ほにゃらら | ふにゃらら |
"""})
    assert len(rows) == 1
    assert rows[0]["subject"] == "その他"
