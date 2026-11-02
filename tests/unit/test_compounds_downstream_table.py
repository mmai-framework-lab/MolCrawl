"""Collecting the downstream probe reports into one row per arm.

The case these cover is one report holding the trained model and its untrained
controls together, which is what --untrained-seeds produces: the trained entry then
has to be found by name rather than by being the only thing beside the fingerprint.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    path = repo_root() / "scripts" / "compounds_downstream_table.py"
    spec = importlib.util.spec_from_file_location("compounds_downstream_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(path: Path, task, metric, molecules, models) -> Path:
    path.write_text(json.dumps({"task": task, "metric": metric,
                                "molecules": molecules, "models": models}))
    return path


def solubility_models(trained_mean, seeds, *, untrained_first=False):
    trained = {"arch": "bert", "size": "small", "mean": trained_mean, "sd": 0.03,
               "adopted_checkpoint": "checkpoint-42800"}
    untrained = {f"bert-small-untrained-seed{i}":
                 {"arch": "bert", "size": "small", "untrained": True,
                  "init_seed": i, "mean": v, "sd": 0.09}
                 for i, v in enumerate(seeds)}
    out = {"ECFP4+linear": {"arch": "none", "mean": 2.0062, "sd": 0.0708}}
    if untrained_first:
        out.update(untrained)
        out["bert-small"] = trained
    else:
        out["bert-small"] = trained
        out.update(untrained)
    return out


def test_the_trained_entry_is_found_whatever_order_json_wrote(mod, tmp_path):
    """One report per arm now carries four model entries, not two."""
    rows = []
    for first in (False, True):
        p = write_report(tmp_path / f"solubility-bert-small-{first}.json", "solubility",
                         "RMSE", 9982,
                         solubility_models(1.2663, [1.6431, 1.7439, 1.7878],
                                           untrained_first=first))
        rows.append(mod.row(p, p))
    assert rows[0][:14] == rows[1][:14]
    assert rows[0][3:5] == ["bert", "small"]
    assert rows[0][6] == "1.2663"


def test_the_improvement_is_signed_by_the_metric_direction(mod, tmp_path):
    """RMSE down is an improvement; ROC-AUC up is. Both must come out positive."""
    p = write_report(tmp_path / "solubility-bert-small.json", "solubility", "RMSE",
                     9982, solubility_models(1.2663, [1.6431, 1.7439, 1.7878]))
    sol = mod.row(p, p)
    assert sol[2] == "lower"
    assert sol[13].startswith("+")

    herg = dict(solubility_models(0.8733, [0.8226, 0.8231, 0.8237]))
    q = write_report(tmp_path / "herg-bert-small.json", "herg", "ROC-AUC", 306893, herg)
    row = mod.row(q, q)
    assert row[2] == "higher"
    assert row[13].startswith("+")


def test_a_missing_untrained_seed_stops_rather_than_averaging_two(mod, tmp_path):
    models = solubility_models(1.2663, [1.6431, 1.7439, 1.7878])
    del models["bert-small-untrained-seed2"]
    p = write_report(tmp_path / "solubility-bert-small.json", "solubility", "RMSE",
                     9982, models)
    with pytest.raises(SystemExit) as caught:
        mod.row(p, p)
    assert "untrained-seed2" in str(caught.value)


def test_the_adopted_checkpoint_is_carried_through(mod, tmp_path):
    """The 45,000-step re-run changed it from checkpoint-14500 to checkpoint-42800."""
    p = write_report(tmp_path / "solubility-bert-small.json", "solubility", "RMSE",
                     9982, solubility_models(1.2663, [1.6431, 1.7439, 1.7878]))
    assert "checkpoint-42800" in mod.row(p, p)


def test_the_trained_and_untrained_values_may_come_from_two_reports(mod, tmp_path):
    """The 2026-10-23 shape, where the controls were a separate set of jobs."""
    trained_only = {k: v for k, v in
                    solubility_models(1.2663, [1.6431, 1.7439, 1.7878]).items()
                    if "-untrained-seed" not in k}
    a = write_report(tmp_path / "a.json", "solubility", "RMSE", 9982, trained_only)
    b = write_report(tmp_path / "b.json", "solubility", "RMSE", 9982,
                     solubility_models(1.2663, [1.6431, 1.7439, 1.7878]))
    row = mod.row(a, b)
    assert row[8:11] == ["1.6431", "1.7439", "1.7878"]
