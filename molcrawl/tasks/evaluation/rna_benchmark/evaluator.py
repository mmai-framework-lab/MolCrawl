"""Arch-agnostic rna_benchmark evaluator.

Scores cell sequences with :meth:`ModelAdapter.score_likelihood` and
reports per-group perplexity + mean log-likelihood.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import CellGroup, load_jsonl, tokens_to_strings
from .metrics import summarise_group

logger = logging.getLogger(__name__)


class RNABenchmarkEvaluator(BaseEvaluator):
    task_name = "rna_benchmark"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        rna_jsonl: Path,
        config: Optional[Dict[str, Any]] = None,
        tracker: Optional[Any] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            handle=handle,
            output_dir=output_dir,
            config=config,
            tracker=tracker,
            experiment_id=experiment_id,
        )
        self.rna_jsonl = Path(rna_jsonl)
        self.datasets: Optional[List[str]] = self.config.get("datasets")
        self.max_cells_per_group: Optional[int] = self.config.get("max_cells_per_group")

    def category(self) -> str:
        return "cell_type_annotation"

    def load_dataset(self) -> Dict[str, CellGroup]:
        groups = load_jsonl(self.rna_jsonl, datasets=self.datasets)
        if self.max_cells_per_group is not None:
            limit = int(self.max_cells_per_group)
            for group in groups.values():
                group.tokens = group.tokens[:limit]
                group.labels = group.labels[:limit]
        return groups

    def run_predictions(self, dataset):
        adapter = self.adapter
        if not adapter.supports("likelihood"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot score likelihoods."
            )
        per_group: Dict[str, np.ndarray] = {}
        for name, group in dataset.items():
            strings = tokens_to_strings(group.tokens)
            out = adapter.score_likelihood(strings)
            per_group[name] = np.asarray(out.log_likelihood, dtype=float)
        return {"log_likelihoods": per_group}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for name, ll in predictions["log_likelihoods"].items():
            summary = summarise_group(ll.tolist())
            for key, value in summary.items():
                metrics[f"{name}.{key}"] = float(value)
        if metrics:
            perplexities = [v for k, v in metrics.items() if k.endswith(".perplexity")]
            if perplexities:
                metrics["mean.perplexity"] = float(np.nanmean(perplexities))
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update({"groups": {name: len(group.tokens) for name, group in dataset.items()}})
        return report
