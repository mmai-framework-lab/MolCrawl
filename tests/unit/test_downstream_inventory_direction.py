"""Which way is better, in the downstream inventory.

The column was the constant "higher". Three of the four metrics in the table are
higher-is-better, so the fourth went unnoticed: nine RMSE rows said a larger
solubility error was the better one, and a reader ranking by that column read the
four compounds sizes backwards.
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
    path = repo_root() / "scripts" / "downstream_inventory_scan.py"
    spec = importlib.util.spec_from_file_location("downstream_inventory_scan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the direction itself ----------------------------------------------------

def test_an_error_metric_is_lower_is_better(mod):
    unknown = []
    assert mod.better_for("RMSE", unknown) == "lower"
    assert mod.better_for("MAE", unknown) == "lower"
    assert unknown == []


def test_a_score_metric_is_higher_is_better(mod):
    unknown = []
    for metric in ("ROC-AUC", "Spearman", "recall@1", "MRR", "accuracy", "R2"):
        assert mod.better_for(metric, unknown) == "higher", metric
    assert unknown == []


def test_the_case_and_the_spacing_do_not_matter(mod):
    unknown = []
    assert mod.better_for("  rmse ", unknown) == "lower"
    assert mod.better_for("roc-auc", unknown) == "higher"
    assert unknown == []


def test_an_undeclared_metric_is_marked_and_named_rather_than_guessed(mod):
    """Guessing "higher" is the defect. A marker sends the reader to the metric."""
    unknown = []
    assert mod.better_for("Tanimoto", unknown) == mod.UNKNOWN_DIRECTION
    assert mod.better_for("Tanimoto", unknown) == mod.UNKNOWN_DIRECTION
    assert unknown == ["Tanimoto"]          # named once, not twice
    assert mod.UNKNOWN_DIRECTION not in ("higher", "lower")


def test_a_missing_metric_is_not_collected_as_an_unknown_name(mod):
    unknown = []
    assert mod.better_for("", unknown) == mod.UNKNOWN_DIRECTION
    assert mod.better_for(None, unknown) == mod.UNKNOWN_DIRECTION
    assert unknown == []


# --- through the compounds reader -------------------------------------------

def write_task(path: Path, task: str, metric: str) -> None:
    path.write_text(json.dumps({
        "task": task, "metric": metric, "molecules": 9982,
        "models": {
            "ECFP4+linear": {"arch": "none", "mean": 2.0062, "sd": 0.0708},
            "bert-small": {
                "arch": "bert", "size": "small", "mean": 1.2663, "sd": 0.0328,
                "adopted_checkpoint": "checkpoint-42800",
                "run_dir": "/x/learning_source_compounds_runs/compounds/"
                           "bert-output/compounds-small-lr1e3-seed42-ext45k",
            },
            "bert-small-untrained-seed0": {
                "arch": "bert", "size": "small", "mean": 1.6431, "sd": 0.09},
        },
    }), encoding="utf-8")


def scan(mod, tmp_path: Path, files: dict) -> list[dict]:
    report = tmp_path / "report"
    (report / "res").mkdir(parents=True)
    for name, (task, metric) in files.items():
        write_task(report / "res" / name, task, metric)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "report_dir": str(report), "strip": "/x/",
        "sources": [{"kind": "compounds_tdc", "modality": "compounds",
                     "dir": "res", "set": "t"}],
    }), encoding="utf-8")
    out = tmp_path / "out.tsv"
    import sys
    argv = sys.argv
    sys.argv = ["downstream_inventory_scan.py", "--config", str(cfg), "--out", str(out)]
    try:
        assert mod.main() == 0
    finally:
        sys.argv = argv
    return list(csv.DictReader(out.read_text(encoding="utf-8").splitlines(),
                               delimiter="\t"))


def test_the_two_compounds_tasks_come_out_with_opposite_directions(mod, tmp_path):
    """The defect in one sentence: these two were both "higher"."""
    rows = scan(mod, tmp_path, {"solubility-bert-small.json": ("solubility", "RMSE"),
                                "herg-bert-small.json": ("herg", "ROC-AUC")})
    got = {r["metric"]: r["better"] for r in rows}
    assert got == {"RMSE": "lower", "ROC-AUC": "higher"}


def test_every_reader_reaches_the_helper(mod, tmp_path):
    """The first fix crashed in read_retrieval, which a compounds-only test misses.

    That reader binds `direction` to the retrieval direction, so a helper of the
    same name was shadowed and the run died on the second source.
    """
    report = tmp_path / "report"
    (report / "ret").mkdir(parents=True)
    (report / "ret" / "r.json").write_text(json.dumps({
        "model_path": "/x/learning_source_rna_runs/gpt2-grid/a-run/ckpt.pt",
        "arch": "gpt2", "size": "small",
        "directions": {"molecule_to_text": {"recall@1": 0.3, "mrr": 0.4,
                                            "random_baseline_recall@1": 0.01,
                                            "queries": 105}},
    }), encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "report_dir": str(report), "strip": "/x/",
        "sources": [{"kind": "molnl_retrieval", "modality": "molnl",
                     "dir": "ret", "set": "t"}],
    }), encoding="utf-8")
    out = tmp_path / "out.tsv"
    import sys
    argv = sys.argv
    sys.argv = ["downstream_inventory_scan.py", "--config", str(cfg), "--out", str(out)]
    try:
        assert mod.main() == 0
    finally:
        sys.argv = argv
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines(),
                               delimiter="\t"))
    assert len(rows) == 1
    assert (rows[0]["metric"], rows[0]["better"]) == ("recall@1", "higher")


def test_the_value_and_the_metric_are_untouched_by_the_fix(mod, tmp_path):
    rows = scan(mod, tmp_path, {"solubility-bert-small.json": ("solubility", "RMSE")})
    assert len(rows) == 1
    assert rows[0]["value"] == "1.2663"
    assert rows[0]["metric"] == "RMSE"
    assert rows[0]["checkpoint"] == "checkpoint-42800"
