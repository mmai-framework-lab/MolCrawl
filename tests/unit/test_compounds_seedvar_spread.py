"""Reading the end-of-run rise off a run's own log, and the spread across seeds.

The measurement is two subtractions, so what can go wrong is not the arithmetic: it
is reading the wrong point as the end, counting a reference run as part of the
spread, or passing over a run whose log is missing without saying so.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load("compounds_seedvar_spread")


def make_run(root: Path, name: str, points, seed=42, max_iters=4674,
             commit="abc123", manifest=True):
    where = root / name
    where.mkdir(parents=True)
    lines = ["iter, train_loss, val_loss"]
    lines += [f"{i}, {v + 0.01:.4f}, {v:.4f}" for i, v in points]
    (where / "logging_20261120_000000.csv").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
    if manifest:
        (where / "run_manifest.json").write_text(json.dumps({
            "seed": {"seed": seed},
            "batch": {"effective_global_batch": 2560},
            "schedule": {"max_iters": max_iters},
            "run": {"git": {"commit": commit}},
        }), encoding="utf-8")
    return where


# --- the series -----------------------------------------------------------------

def test_the_header_is_not_read_as_a_point(mod, tmp_path):
    where = make_run(tmp_path, "r", [(0, 6.5), (50, 1.0)])
    points, used = mod.series(str(where))
    assert points == [(0, 6.5), (50, 1.0)]
    assert used == ["logging_20261120_000000.csv"]


def test_a_run_without_a_log_is_empty_not_an_error(mod, tmp_path):
    (tmp_path / "empty").mkdir()
    points, used = mod.series(str(tmp_path / "empty"))
    assert points == [] and used == []


# --- the two numbers ------------------------------------------------------------

def test_the_rise_is_the_end_minus_the_minimum(mod, tmp_path):
    where = make_run(tmp_path, "r", [(0, 6.5), (2050, 0.5496), (4674, 1.0756)])
    row = mod.row_for("x", "spread", str(where))
    assert row["best_value"] == "0.549600"
    assert row["best_iter"] == 2050
    assert row["final_value"] == "1.075600"
    assert row["final_minus_best"] == "+0.526000"


def test_the_end_is_the_last_point_not_the_largest(mod, tmp_path):
    # A run that peaked mid-way and came back down must not report the peak as the end.
    where = make_run(tmp_path, "r", [(0, 6.5), (100, 0.5), (200, 9.0), (300, 0.7)])
    row = mod.row_for("x", "spread", str(where))
    assert row["final_iter"] == 300
    assert row["final_minus_best"] == "+0.200000"


def test_the_position_is_against_the_planned_length(mod, tmp_path):
    # Not against the last point reached: a run cut short would otherwise read 100 %.
    where = make_run(tmp_path, "r", [(0, 6.5), (2337, 0.5)], max_iters=4674)
    row = mod.row_for("x", "spread", str(where))
    assert row["best_fraction_of_budget"] == "0.5000"


def test_facts_come_from_the_manifest(mod, tmp_path):
    where = make_run(tmp_path, "r", [(0, 1.0)], seed=17, commit="deadbee")
    row = mod.row_for("x", "spread", str(where))
    assert row["seed"] == 17 and row["commit"] == "deadbee" and row["batch"] == 2560


def test_a_missing_manifest_leaves_the_facts_empty_and_still_reads_the_series(mod, tmp_path):
    where = make_run(tmp_path, "r", [(0, 6.5), (50, 1.0)], manifest=False)
    row = mod.row_for("x", "spread", str(where))
    assert row["seed"] is None
    assert row["final_minus_best"] == "+0.000000"  # the end IS the minimum here


def test_a_run_with_no_series_returns_nothing_to_report(mod, tmp_path):
    (tmp_path / "empty").mkdir()
    assert mod.row_for("x", "spread", str(tmp_path / "empty")) is None


# --- the spread, end to end ------------------------------------------------------

def test_only_the_spread_runs_count_and_a_missing_one_is_named(mod, tmp_path, capsys):
    make_run(tmp_path, "s42", [(0, 6.5), (2000, 0.55), (4674, 1.05)], seed=42)
    make_run(tmp_path, "s1", [(0, 6.5), (2000, 0.55), (4674, 0.70)], seed=1)
    make_run(tmp_path, "ref", [(0, 6.5), (2900, 0.55), (5850, 0.65)], seed=42)
    (tmp_path / "s17").mkdir()
    out = tmp_path / "rt1-probe.tsv"
    rc = mod.main(["--run", f"s42={tmp_path / 's42'}",
                   "--run", f"s1={tmp_path / 's1'}",
                   "--run", f"s17={tmp_path / 's17'}",
                   "--reference", f"ref={tmp_path / 'ref'}",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 1                      # the missing run makes it non-zero
    assert "no evaluation series for s17" in printed
    # The reference run's +0.10 must not widen the spread, which is +0.15 to +0.50.
    assert "+0.1500 to +0.5000" in printed
    assert "width 0.3500" in printed
    assert "ref: +0.1000 -- OUTSIDE" in printed
    body = out.read_text(encoding="utf-8")
    assert body.splitlines()[0] == "#! report-tsv 1"
    assert "reference" in body and "spread" in body


def test_a_label_without_a_path_is_refused(mod):
    with pytest.raises(SystemExit):
        mod.pairs(["justalabel"])
