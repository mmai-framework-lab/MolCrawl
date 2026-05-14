"""Arch-agnostic rna_benchmark evaluator.

Scores cell sequences with :meth:`ModelAdapter.score_likelihood` and
reports per-group perplexity + mean log-likelihood.

足固め upgrade adds:

- reproducible per-group subsampling
- bootstrap 95 % CI on per-group perplexity
- per-cell predictions log (jsonl + narrative TXT)
- direct int-list pass-through to the adapter (no string round-trip),
  enabling rna BERT / rnaformer to score raw token ids straight from
  the parquet pipeline output
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import CellGroup, load_jsonl, subsample_groups
from .metrics import bootstrap_perplexity_ci, summarise_group
from .predictions_log import write_predictions

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
        self.cells_per_group: Optional[int] = self.config.get(
            "cells_per_group", self.config.get("max_cells_per_group")
        )
        self.seed: int = int(self.config.get("seed", 42))
        self.bootstrap_samples: int = int(self.config.get("bootstrap_samples", 100))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 6)
        )

    def category(self) -> str:
        return "cell_type_annotation"

    def load_dataset(self) -> Dict[str, CellGroup]:
        groups = load_jsonl(self.rna_jsonl, datasets=self.datasets)
        if self.cells_per_group is not None:
            groups = subsample_groups(
                groups,
                cells_per_group=int(self.cells_per_group),
                seed=self.seed,
            )
        return groups

    def run_predictions(self, dataset):
        adapter = self.adapter
        if not adapter.supports("likelihood"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot score likelihoods."
            )
        per_cell: Dict[str, Dict[str, list]] = {}
        for name, group in dataset.items():
            logger.info(
                "Scoring %d cells for tissue=%r (mean tokens=%.0f)",
                len(group.tokens),
                name,
                float(np.mean(group.token_counts)) if group.token_counts else 0.0,
            )
            out = adapter.score_likelihood(group.tokens)
            per_cell[name] = {
                "log_likelihood": [float(v) for v in out.log_likelihood],
                "token_count": list(group.token_counts),
                "num_scored_tokens": [int(v) for v in (out.num_tokens or [])],
            }
        return {"per_cell": per_cell}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        bootstrap_payload: Dict[str, Dict[str, float]] = {}
        per_cell = predictions["per_cell"]
        all_ll: List[float] = []
        for name, data in per_cell.items():
            ll = data["log_likelihood"]
            summary = summarise_group(ll)
            for key, value in summary.items():
                metrics[f"{name}.{key}"] = float(value) if value is not None else float("nan")
            ci_lo, ci_hi = bootstrap_perplexity_ci(
                ll,
                n_boot=self.bootstrap_samples,
                seed=self.seed,
            )
            bootstrap_payload[name] = {"ci_lo": ci_lo, "ci_hi": ci_hi}
            all_ll.extend(ll)

        if metrics:
            perplexities = [
                v for k, v in metrics.items() if k.endswith(".perplexity")
            ]
            if perplexities:
                metrics["mean.perplexity"] = float(np.nanmean(perplexities))
            if all_ll:
                pooled = summarise_group(all_ll)
                metrics["all.mean_log_likelihood"] = pooled["mean_log_likelihood"]
                metrics["all.perplexity"] = pooled["perplexity"]
                bootstrap_payload["all"] = dict(
                    zip(
                        ("ci_lo", "ci_hi"),
                        bootstrap_perplexity_ci(
                            all_ll, n_boot=self.bootstrap_samples, seed=self.seed
                        ),
                    )
                )

        self._last_bootstrap_ci = bootstrap_payload
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        artefacts = write_predictions(
            output_dir=self.output_dir,
            per_cell=predictions["per_cell"],
            arch=self.handle.arch,
            preview_count=self.predictions_preview_count,
        )
        report.update(
            {
                "groups": {name: len(g.tokens) for name, g in dataset.items()},
                "seed": self.seed,
                "bootstrap_ci_95": getattr(self, "_last_bootstrap_ci", {}),
                "artefacts": artefacts,
                "notes": (
                    "rna_benchmark scores cells via PLL on the configured MLM "
                    "(bert / rnaformer). Tokens are passed straight to the adapter "
                    "(no string round-trip), so per-cell perplexity is comparable "
                    "across architectures that share the gene vocabulary. "
                    "Per-group bootstrap CIs use 100 resamples by default."
                ),
            }
        )
        return report
