"""Arch-agnostic ProteinGym evaluator.

Implements the standard ProteinGym protocol — score =
LL(mutant) − LL(wildtype), correlated against the experimental DMS
score. The足固め upgrade promotes the continuous ranking metrics
(Spearman / Pearson) to primary and adds bootstrap 95 % CIs,
per-class (functional / non-functional) score-distribution
diagnostics, and a per-variant prediction dump (JSONL +
quantile-sampled narrative) so each run is inspectable without
re-running.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_proteingym
from .metrics import (
    bootstrap_correlation_ci,
    correlation_metrics,
    optional_binary_metrics,
    score_distribution_stats,
)
from .predictions_log import write_predictions
from .splits import sample_proteingym

logger = logging.getLogger(__name__)

_MIN_ROWS_FOR_CORRELATION = 10


class ProteinGymEvaluator(BaseEvaluator):
    """Zero-shot variant fitness prediction via likelihood differences."""

    task_name = "proteingym"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        proteingym_path: Path,
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
        self.proteingym_path = Path(proteingym_path)
        self.context_length: int = int(self.config.get("context_length", 1024))

    def category(self) -> str:
        return "variant_effect"

    def load_dataset(self) -> pd.DataFrame:
        df = load_proteingym(self.proteingym_path)

        n_examples = self.config.get("n_examples")
        max_examples = self.config.get("max_examples")
        if n_examples is None and max_examples is not None:
            n_examples = int(max_examples)
            logger.warning(
                "ProteinGymEvaluator: max_examples=%s deprecated; "
                "re-interpreting as n_examples=%d (no structural change).",
                max_examples,
                n_examples,
            )

        stratify_bin = bool(self.config.get("stratify_bin", True))
        seed = int(self.config.get("seed", 42))

        sampled = sample_proteingym(
            df,
            n_examples=n_examples,
            stratify_bin=stratify_bin,
            seed=seed,
        )
        self._last_sampling = {
            "n_examples": n_examples,
            "stratify_bin": stratify_bin,
            "seed": seed,
            "total_rows_in_file": int(len(df)),
        }
        return sampled

    def run_predictions(self, dataset: pd.DataFrame) -> Dict[str, np.ndarray]:
        adapter = self.adapter
        if not adapter.supports("likelihood"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot score likelihoods, "
                "which ProteinGym requires."
            )
        ref_out = adapter.score_likelihood(
            dataset["wildtype_sequence"].astype(str).tolist(),
            context_length=self.context_length,
        )
        mut_out = adapter.score_likelihood(
            dataset["mutated_sequence"].astype(str).tolist(),
            context_length=self.context_length,
        )
        ref_ll = np.asarray(ref_out.log_likelihood, dtype=float)
        mut_ll = np.asarray(mut_out.log_likelihood, dtype=float)
        return {
            "scores": mut_ll - ref_ll,
            "reference_ll": ref_ll,
            "mutated_ll": mut_ll,
        }

    def compute_metrics(
        self, dataset: pd.DataFrame, predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        scores = predictions["scores"]
        dms = dataset["DMS_score"].to_numpy(dtype=float)
        ref_ll = predictions["reference_ll"]
        mut_ll = predictions["mutated_ll"]

        if len(dms) < _MIN_ROWS_FOR_CORRELATION:
            logger.warning(
                "ProteinGymEvaluator: skipping correlation metrics (%d rows < %d)",
                len(dms),
                _MIN_ROWS_FOR_CORRELATION,
            )
            self._last_bootstrap = {}
            self._last_score_distribution = {}
            return {}

        metrics = correlation_metrics(dms, scores)

        bin_labels: Optional[np.ndarray] = None
        if "DMS_bin_score" in dataset.columns:
            bin_series = dataset["DMS_bin_score"].dropna()
            if not bin_series.empty:
                bin_labels = dataset["DMS_bin_score"].to_numpy()
                extra = optional_binary_metrics(
                    bin_series.astype(int).to_numpy(),
                    scores[dataset["DMS_bin_score"].notna().to_numpy()],
                )
                metrics.update(extra)

        n_boot = int(self.config.get("bootstrap_samples", 200))
        self._last_bootstrap = bootstrap_correlation_ci(
            dms, scores, n_boot=n_boot, seed=int(self.config.get("seed", 42))
        )
        self._last_score_distribution = score_distribution_stats(
            dms, ref_ll, mut_ll, scores, bin_labels=bin_labels
        )
        return metrics

    def build_report(
        self,
        metrics: Dict[str, float],
        dataset: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        report = super().build_report(metrics, dataset, predictions)

        bootstrap_ci = getattr(self, "_last_bootstrap", None) or {}
        bootstrap_payload: Dict[str, Dict[str, float]] = {}
        for key, (lo, hi) in bootstrap_ci.items():
            bootstrap_payload[key] = {"ci_lo": float(lo), "ci_hi": float(hi)}

        sampling = getattr(self, "_last_sampling", None)
        score_distribution = getattr(self, "_last_score_distribution", None)

        preview_count = int(self.config.get("predictions_preview_count", 20))
        prediction_paths = write_predictions(
            output_dir=self.output_dir,
            dataset=dataset,
            predictions=predictions,
            score_distribution=score_distribution,
            sampling=sampling,
            arch=self.handle.arch,
            modality=self.handle.modality,
            preview_count=preview_count,
        )

        report.update(
            {
                "num_variants": int(len(dataset)),
                "context_length": self.context_length,
                "sampling": sampling,
                "assay_csv": str(self.proteingym_path),
                "dms_summary": {
                    "min": float(dataset["DMS_score"].min()) if len(dataset) else None,
                    "max": float(dataset["DMS_score"].max()) if len(dataset) else None,
                    "median": float(dataset["DMS_score"].median()) if len(dataset) else None,
                    "mean": float(dataset["DMS_score"].mean()) if len(dataset) else None,
                },
                "score_distribution": score_distribution,
                "bootstrap_ci_95": bootstrap_payload,
                "artefacts": prediction_paths,
                "notes": (
                    "Score is LL(mutant) − LL(wildtype); positive Spearman "
                    "with DMS_score means the model's ranking of variants "
                    "agrees with the experimental fitness order. ProteinGym "
                    "reports are per-assay; running across the full 217-assay "
                    "set requires an outer workflow that iterates this "
                    "evaluator over each CSV and aggregates the per-assay "
                    "Spearman values."
                ),
            }
        )
        return report
