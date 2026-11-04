"""Stitching the three compounds stages into one train/valid curve table.

The cases that matter are the ones the stitch exists for: each stage's own
history supplying its own steps, a later stage's restored copy being checked
rather than trusted, and a step that carries only one of the two splits.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    path = repo_root() / "scripts" / "compounds_trainval_curve_tsv.py"
    spec = importlib.util.spec_from_file_location("compounds_trainval_curve_tsv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(root: Path, name: str, final_step: int, history, extra_ckpts=()) -> Path:
    """A run directory with a checkpoint at final_step carrying this history."""
    run = root / name
    for step in (*extra_ckpts, final_step):
        (run / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
    state = {"global_step": final_step, "log_history": history}
    (run / f"checkpoint-{final_step}" / "trainer_state.json").write_text(
        json.dumps(state), encoding="utf-8")
    return run


def points(lo, hi, *, train=0.5, valid=0.4, step=100):
    """train and valid entries at every `step` from lo to hi, as HF writes them."""
    out = []
    for s in range(lo, hi + 1, step):
        out.append({"step": s, "loss": train + s / 1e6})
        out.append({"step": s, "eval_loss_mask": valid + s / 1e6})
    return out


def build_arm(root: Path, base: str, *, tail=None):
    """The three stage directories of one arm, with agreeing restored histories."""
    stage1 = points(100, 15_000)
    stage2 = points(15_100, 30_000)
    stage3 = tail if tail is not None else points(30_100, 45_000)
    write_run(root, base, 15_000, stage1)
    write_run(root, base + "-ext30k", 30_000, stage1 + stage2)
    write_run(root, base + "-ext45k", 45_000, stage1 + stage2 + stage3)


def read_tsv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@pytest.fixture
def one_arm(mod, tmp_path, monkeypatch):
    """Only the small arm, so a test does not have to build four trees."""
    monkeypatch.setattr(mod, "ARMS", [("small", "1e-3", "compounds-small-lr1e3-seed42")])
    return tmp_path


def run(mod, root: Path, out: Path):
    return mod.main(["--bert-root", str(root), "--out", str(out)])


def test_columns_match_the_2026_10_20_table(mod):
    assert mod.COLUMNS == ["arch", "size", "lr", "seed", "step", "split", "value", "source"]


def test_every_step_carries_both_splits(mod, one_arm, capsys):
    build_arm(one_arm, "compounds-small-lr1e3-seed42")
    out = one_arm / "curve.tsv"
    assert run(mod, one_arm, out) == 0
    rows = read_tsv(out)
    assert len(rows) == 450 * 2
    steps = {int(r["step"]) for r in rows}
    assert min(steps) == 100 and max(steps) == 45_000 and len(steps) == 450
    assert {r["split"] for r in rows} == {"train", "valid"}
    assert "報告すべき不一致 0 件" in capsys.readouterr().out


def test_lr_is_the_stage1_peak_not_the_later_constant(mod, one_arm):
    build_arm(one_arm, "compounds-small-lr1e3-seed42")
    out = one_arm / "curve.tsv"
    run(mod, one_arm, out)
    assert {r["lr"] for r in read_tsv(out)} == {"1e-3"}
    assert {r["seed"] for r in read_tsv(out)} == {"42"}


def test_each_stage_supplies_its_own_steps(mod, one_arm):
    build_arm(one_arm, "compounds-small-lr1e3-seed42")
    out = one_arm / "curve.tsv"
    run(mod, one_arm, out)
    source = {int(r["step"]): r["source"] for r in read_tsv(out) if r["split"] == "valid"}
    assert source[15_000].startswith("compounds-small-lr1e3-seed42/checkpoint-15000/")
    assert source[15_100].startswith("compounds-small-lr1e3-seed42-ext30k/")
    assert source[30_000].startswith("compounds-small-lr1e3-seed42-ext30k/")
    assert source[30_100].startswith("compounds-small-lr1e3-seed42-ext45k/")


def test_value_comes_from_the_stage_that_trained_the_step(mod, one_arm):
    """A later stage's restored copy does not get to overwrite an earlier stage."""
    base = "compounds-small-lr1e3-seed42"
    stage1 = points(100, 15_000)
    stage2 = points(15_100, 30_000)
    stage3 = points(30_100, 45_000)
    altered = [dict(e) for e in stage1]
    for entry in altered:
        if "eval_loss_mask" in entry and entry["step"] == 5_000:
            entry["eval_loss_mask"] = 9.0
    write_run(one_arm, base, 15_000, stage1)
    write_run(one_arm, base + "-ext30k", 30_000, stage1 + stage2)
    write_run(one_arm, base + "-ext45k", 45_000, altered + stage2 + stage3)
    out = one_arm / "curve.tsv"
    run(mod, one_arm, out)
    row = next(r for r in read_tsv(out) if int(r["step"]) == 5_000 and r["split"] == "valid")
    assert row["value"] != "9.000000"


def test_a_disagreeing_restored_history_is_reported(mod, one_arm, capsys):
    base = "compounds-small-lr1e3-seed42"
    stage1 = points(100, 15_000)
    stage2 = points(15_100, 30_000)
    stage3 = points(30_100, 45_000)
    altered = [dict(e) for e in stage1]
    for entry in altered:
        if "eval_loss_mask" in entry and entry["step"] == 5_000:
            entry["eval_loss_mask"] = 9.0
    write_run(one_arm, base, 15_000, stage1)
    write_run(one_arm, base + "-ext30k", 30_000, stage1 + stage2)
    write_run(one_arm, base + "-ext45k", 45_000, altered + stage2 + stage3)
    run(mod, one_arm, out := one_arm / "curve.tsv")
    assert out.exists()
    text = capsys.readouterr().out
    assert "最大差" in text
    assert "報告すべき不一致 0 件" not in text


def test_a_step_with_only_one_split_says_so_in_source(mod, one_arm):
    base = "compounds-small-lr1e3-seed42"
    tail = [e for e in points(30_100, 45_000)
            if not (e["step"] == 45_000 and "eval_loss_mask" in e)]
    build_arm(one_arm, base, tail=tail)
    run(mod, one_arm, out := one_arm / "curve.tsv")
    rows = [r for r in read_tsv(out) if int(r["step"]) == 45_000]
    assert [r["split"] for r in rows] == ["train"]
    assert "valid not recorded at this step" in rows[0]["source"]


def test_a_missing_eval_point_is_reported(mod, one_arm, capsys):
    base = "compounds-small-lr1e3-seed42"
    tail = [e for e in points(30_100, 45_000) if e["step"] != 40_000]
    build_arm(one_arm, base, tail=tail)
    run(mod, one_arm, one_arm / "curve.tsv")
    text = capsys.readouterr().out
    assert "40000" in text and "点が無い" in text


def test_the_joins_are_printed_with_both_sides(mod, one_arm, capsys):
    build_arm(one_arm, "compounds-small-lr1e3-seed42")
    run(mod, one_arm, one_arm / "curve.tsv")
    text = capsys.readouterr().out
    assert "つなぎ 15,000 -> 15,100" in text
    assert "つなぎ 30,000 -> 30,100" in text


def test_a_duplicated_step_is_reported(mod, one_arm, capsys):
    base = "compounds-small-lr1e3-seed42"
    tail = points(30_100, 45_000) + [{"step": 45_000, "eval_loss_mask": 0.44}]
    build_arm(one_arm, base, tail=tail)
    run(mod, one_arm, one_arm / "curve.tsv")
    text = capsys.readouterr().out
    assert "同じ step が 2 回" in text


def test_the_retained_checkpoints_are_listed(mod, one_arm, capsys):
    base = "compounds-small-lr1e3-seed42"
    build_arm(one_arm, base)
    (one_arm / base / "checkpoint-14000").mkdir(exist_ok=True)
    run(mod, one_arm, one_arm / "curve.tsv")
    assert "[14000, 15000]" in capsys.readouterr().out


def test_a_run_with_no_checkpoint_stops_rather_than_writing_a_short_table(mod, one_arm):
    (one_arm / "compounds-small-lr1e3-seed42").mkdir()
    with pytest.raises(SystemExit):
        run(mod, one_arm, one_arm / "curve.tsv")


def test_the_file_is_tab_separated_with_crlf_and_no_bom(mod, one_arm):
    build_arm(one_arm, "compounds-small-lr1e3-seed42")
    run(mod, one_arm, out := one_arm / "curve.tsv")
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.split(b"\r\n")[0] == b"arch\tsize\tlr\tseed\tstep\tsplit\tvalue\tsource"
