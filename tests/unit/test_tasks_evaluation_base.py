"""Unit tests for the Phase 0 evaluation foundation.

These tests only exercise the architecture-agnostic parts of
``molcrawl.tasks.evaluation._base`` + ``_adapters`` so they can run in a
minimal environment without the full deep-learning stack installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pytest

from molcrawl.tasks.evaluation._base import (
    BaseEvaluator,
    ModelAdapter,
    ModelHandle,
    default_registry,
)
from molcrawl.tasks.evaluation._base.model_adapter import (
    LikelihoodOutput,
    available_adapters,
    build_adapter,
    register_adapter,
)


class DummyAdapter(ModelAdapter):
    """Adapter that echoes log-likelihoods from a lookup table."""

    def __init__(self, handle: ModelHandle):
        super().__init__(handle)
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def score_likelihood(
        self, inputs: Sequence[str], context_length: int = 512, **_: Any
    ) -> LikelihoodOutput:
        table: Dict[str, float] = self.handle.extras.get("likelihoods", {})
        scores = [float(table.get(s, 0.0)) for s in inputs]
        return LikelihoodOutput(log_likelihood=scores, num_tokens=[len(s) for s in inputs])


register_adapter("dummy", DummyAdapter)


class EchoEvaluator(BaseEvaluator):
    task_name = "echo"

    def category(self) -> str:
        return "variant_effect"

    def load_dataset(self):
        return [
            {"seq": "AAA", "label": 1},
            {"seq": "CCC", "label": 0},
        ]

    def run_predictions(self, dataset):
        out = self.adapter.score_likelihood([row["seq"] for row in dataset])
        return out.log_likelihood

    def compute_metrics(self, dataset, predictions):
        labels = [row["label"] for row in dataset]
        preds = [1 if p > 0 else 0 for p in predictions]
        return {
            "accuracy": default_registry.compute("accuracy", labels, preds),
        }


def test_default_registry_has_core_metrics():
    names = default_registry.list()
    for required in ("perplexity", "accuracy", "rmse", "validity"):
        assert required in names


def test_classification_metrics_compute():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0]
    acc = default_registry.compute("accuracy", y_true, y_pred)
    f1 = default_registry.compute("f1_binary", y_true, y_pred)
    assert acc == pytest.approx(0.8)
    assert 0.0 <= f1 <= 1.0


def test_regression_metrics_compute():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    assert default_registry.compute("rmse", y_true, y_pred) == pytest.approx(0.0)
    assert default_registry.compute("r2", y_true, y_pred) == pytest.approx(1.0)


def test_adapter_registry_reports_dummy():
    assert "dummy" in available_adapters()


def test_base_evaluator_runs_end_to_end(tmp_path: Path):
    handle = ModelHandle(
        arch="dummy",
        modality="genome_sequence",
        model_path="unused",
        tokenizer_path=None,
        extras={"likelihoods": {"AAA": 1.0, "CCC": -1.0}},
    )

    evaluator = EchoEvaluator(
        handle=handle,
        output_dir=tmp_path / "report",
    )
    result = evaluator.run()

    assert result.task == "echo"
    assert result.metrics["accuracy"] == pytest.approx(1.0)
    metrics_path = Path(result.report_paths["json"])
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["arch"] == "dummy"
    assert payload["modality"] == "genome_sequence"
    assert payload["category"] == "variant_effect"

    md_path = Path(result.report_paths["markdown"])
    assert md_path.exists()
    assert "# Evaluation report" in md_path.read_text()


def test_build_adapter_errors_on_unknown_arch():
    handle = ModelHandle(arch="not_registered", modality="x", model_path="y")
    with pytest.raises(KeyError):
        build_adapter(handle)
