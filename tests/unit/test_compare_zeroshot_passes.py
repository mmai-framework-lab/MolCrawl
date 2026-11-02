"""Holding the 2026-11-02 scoring against the one of 2026-10-30.

The reported fold table is built from the earlier aggregates, so the new pass
has to be shown to reproduce them rather than replace them. The rows of
2026-10-30 carry no identifier -- which is why they are being re-scored -- so
they can only be joined on the line number, and that join has to refuse to
report agreement when the two files are not in the same order.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "compare_zeroshot_passes.py"
_spec = importlib.util.spec_from_file_location("_compare_passes", _SRC)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def _write(d, subset, aurocs, rows, score_key, with_id, step=99000):
    d.mkdir(parents=True, exist_ok=True)
    body = {"checkpoint_step": step}
    body.update(aurocs)
    json.dump(body, open(d / f"window-{subset}.json", "w"))
    lines = []
    for i, (chrom, label, score) in enumerate(rows):
        r = {"chrom": chrom, "fold": f"chr{chrom}", "label_pathogenic": label,
             score_key: score}
        if with_id:
            r = {"vcv_id": f"V{i:02d}", **r}
        lines.append(json.dumps(r))
    (d / f"window-{subset}-predictions.jsonl").write_text("\n".join(lines) + "\n")


AUROCS = {"auroc_all": 0.65, "auroc_three": 0.65, "auroc_chr21": 0.60,
          "auroc_chr22": 0.70, "auroc_chrX": 0.65}
ROWS = [("21", 1, -0.5), ("21", 0, -0.7), ("22", 1, -0.2), ("X", 0, -0.9)]


def _pair(tmp_path, new_rows=None, new_aurocs=None, new_step=99000,
          old_with_id=False):
    new, old = tmp_path / "new", tmp_path / "old"
    _write(new, "global_random_seed1", new_aurocs or AUROCS,
           new_rows or ROWS, "score_window", True, step=new_step)
    _write(old, "global_random_seed1", AUROCS, ROWS, "score", old_with_id)
    return new, old


def _run(new, old, extra=()):
    old_argv = sys.argv
    sys.argv = ["x", "--new", str(new), "--old", str(old), "--tolerance", "0",
                *extra]
    try:
        cp.main()
    finally:
        sys.argv = old_argv


def test_an_identical_pass_passes(tmp_path, capsys):
    new, old = _pair(tmp_path)
    _run(new, old)
    out = capsys.readouterr().out
    assert "問題なし" in out
    assert "4 行" in out


def test_rows_without_identifiers_are_joined_on_the_line_number(tmp_path, capsys):
    new, old = _pair(tmp_path)
    _run(new, old)
    assert "行番号" in capsys.readouterr().out


def test_identifiers_are_used_when_the_old_pass_has_them(tmp_path, capsys):
    new, old = _pair(tmp_path, old_with_id=True)
    _run(new, old)
    assert "vcv_id" in capsys.readouterr().out


def test_a_moved_aggregate_fails_the_comparison(tmp_path):
    moved = dict(AUROCS, auroc_chr22=0.70001)
    new, old = _pair(tmp_path, new_aurocs=moved)
    with pytest.raises(SystemExit):
        _run(new, old)


def test_a_changed_per_variant_value_fails_the_comparison(tmp_path):
    rows = list(ROWS)
    rows[2] = ("22", 1, -0.2000001)
    new, old = _pair(tmp_path, new_rows=rows)
    with pytest.raises(SystemExit):
        _run(new, old)


def test_a_different_adopted_checkpoint_fails_the_comparison(tmp_path):
    new, old = _pair(tmp_path, new_step=151000)
    with pytest.raises(SystemExit):
        _run(new, old)


def test_rows_out_of_order_are_not_reported_as_agreement(tmp_path):
    """Same values, different order: the line-number join must notice."""
    rows = [ROWS[1], ROWS[0], ROWS[2], ROWS[3]]
    new, old = _pair(tmp_path, new_rows=rows)
    with pytest.raises(SystemExit):
        _run(new, old)


def test_a_tolerance_can_be_given_but_zero_is_the_default(tmp_path, capsys):
    rows = list(ROWS)
    rows[2] = ("22", 1, -0.2000001)
    new, old = _pair(tmp_path, new_rows=rows)
    old_argv = sys.argv
    sys.argv = ["x", "--new", str(new), "--old", str(old), "--tolerance", "1e-5"]
    try:
        cp.main()
    finally:
        sys.argv = old_argv
    assert "問題なし" in capsys.readouterr().out
