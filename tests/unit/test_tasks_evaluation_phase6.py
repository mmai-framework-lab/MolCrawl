"""Unit tests for the Phase 6 snapshot aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molcrawl.tasks.evaluation._snapshot import (
    build_snapshot,
    collect_results,
    diff_with_previous,
    write_snapshot,
)


def _write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sample_payload(task: str, arch: str, modality: str, metrics: dict, ts: str) -> dict:
    return {
        "generated_at": ts,
        "task": task,
        "modality": modality,
        "arch": arch,
        "category": "property_prediction",
        "metrics": metrics,
    }


def test_collect_and_build_snapshot(tmp_path: Path):
    root = tmp_path / "eval"
    _write_metrics(
        root / "moleculenet/bbbp/metrics.json",
        _sample_payload("moleculenet", "chemberta2", "compounds", {"auroc": 0.8}, "2026-04-20T00:00:00Z"),
    )
    _write_metrics(
        root / "clinvar/metrics.json",
        _sample_payload("clinvar", "gpt2", "genome_sequence", {"f1_binary": 0.6}, "2026-04-20T00:00:00Z"),
    )
    entries = collect_results(root)
    assert len(entries) == 2

    snapshot = build_snapshot(entries)
    assert len(snapshot["runs"]) == 2
    modality_arch_task = {(r["modality"], r["arch"], r["task"]) for r in snapshot["runs"]}
    assert ("compounds", "chemberta2", "moleculenet") in modality_arch_task
    assert ("genome_sequence", "gpt2", "clinvar") in modality_arch_task


def test_build_snapshot_keeps_latest(tmp_path: Path):
    root = tmp_path / "eval"
    _write_metrics(
        root / "moleculenet/bbbp/run1/metrics.json",
        _sample_payload("moleculenet", "chemberta2", "compounds", {"auroc": 0.7}, "2026-04-20T00:00:00Z"),
    )
    _write_metrics(
        root / "moleculenet/bbbp/run2/metrics.json",
        _sample_payload("moleculenet", "chemberta2", "compounds", {"auroc": 0.9}, "2026-04-21T00:00:00Z"),
    )
    entries = collect_results(root)
    snapshot = build_snapshot(entries)
    auroc_values = [r["metrics"].get("auroc") for r in snapshot["runs"]]
    assert auroc_values == [0.9]


def test_diff_with_previous():
    previous = {
        "runs": [
            {
                "task": "moleculenet",
                "modality": "compounds",
                "arch": "chemberta2",
                "metrics": {"auroc": 0.8},
            }
        ]
    }
    current = {
        "runs": [
            {
                "task": "moleculenet",
                "modality": "compounds",
                "arch": "chemberta2",
                "metrics": {"auroc": 0.85},
            }
        ]
    }
    diff = diff_with_previous(current, previous)
    assert diff["compounds/chemberta2/moleculenet"]["auroc"] == pytest.approx(0.05)


def test_write_snapshot_creates_both_files(tmp_path: Path):
    snapshot = {
        "generated_at": "2026-04-22T00:00:00Z",
        "runs": [
            {
                "task": "moleculenet",
                "modality": "compounds",
                "arch": "chemberta2",
                "category": "property_prediction",
                "metrics": {"auroc": 0.8},
                "source": "x",
            }
        ],
    }
    paths = write_snapshot(snapshot, output_dir=tmp_path, name="20260422")
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    md = Path(paths["markdown"]).read_text()
    assert "moleculenet" in md
