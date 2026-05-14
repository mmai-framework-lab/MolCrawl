"""Architecture-agnostic COSMIC evaluator.

Parallels the ClinVar evaluator: scores ``LL(ref) − LL(var)`` for each
variant and reports threshold-free ranking metrics (AUROC / AUPRC) as
primary signal, with the F1-optimal threshold-based metrics retained as
a secondary block (same-set-fitted, per the zero-shot variant-effect
convention).

What 足固め mode adds on top of the basic flow:

- Reproducible class-balanced sampling (optionally tier-stratified)
  via :func:`splits.sample_cosmic` so the dataset's tier-3-dominated
  imbalance does not silently bias the metric estimate.
- 95 % bootstrap CIs for every metric, computed on the same sample at
  the same threshold so the CI reflects sampling variance only.
- Per-class score-distribution stats and per-variant predictions log
  (JSONL + narrative TXT) so the output is inspectable without
  re-running the model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle, default_registry

from .data_preparation import load_cosmic
from .metrics import (
    bootstrap_binary_ci,
    confusion_summary,
    find_optimal_f1_threshold,
    score_distribution_stats,
    sensitivity_specificity,
)
from .predictions_log import write_predictions
from .splits import sample_cosmic

logger = logging.getLogger(__name__)

_MIN_PER_CLASS_FOR_THRESHOLD_METRICS = 10
_MIN_TOTAL_FOR_RANKING_METRICS = 20


class CosmicEvaluator(BaseEvaluator):
    task_name = "cosmic"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        cosmic_path: Path,
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
        self.cosmic_path = Path(cosmic_path)
        self.context_length: int = int(self.config.get("context_length", 512))
        self.label_column: str = str(self.config.get("label_column", "FATHMM_PREDICTION"))

    def category(self) -> str:
        return "variant_effect"

    def load_dataset(self) -> pd.DataFrame:
        df = load_cosmic(self.cosmic_path, label_column=self.label_column)

        n_per_class = self.config.get("n_per_class")
        max_examples = self.config.get("max_examples")
        if n_per_class is None and max_examples is not None:
            n_per_class = max(1, int(max_examples) // 2)
            logger.warning(
                "CosmicEvaluator: max_examples=%s is deprecated; re-interpreting "
                "as n_per_class=%d (class-balanced).",
                max_examples,
                n_per_class,
            )

        stratify_tier = bool(self.config.get("stratify_tier", True))
        seed = int(self.config.get("seed", 42))

        sampled = sample_cosmic(
            df,
            n_per_class=n_per_class,
            stratify_tier=stratify_tier,
            seed=seed,
        )
        self._last_sampling = {
            "n_per_class": n_per_class,
            "stratify_tier": stratify_tier,
            "seed": seed,
            "total_rows_in_file": int(len(df)),
        }
        return sampled

    def run_predictions(self, dataset: pd.DataFrame) -> Dict[str, np.ndarray]:
        adapter = self.adapter
        if not adapter.supports("likelihood"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot score likelihoods."
            )
        ref_out = adapter.score_likelihood(
            dataset["reference_sequence"].astype(str).tolist(),
            context_length=self.context_length,
        )
        var_out = adapter.score_likelihood(
            dataset["variant_sequence"].astype(str).tolist(),
            context_length=self.context_length,
        )
        ref_ll = np.asarray(ref_out.log_likelihood, dtype=float)
        var_ll = np.asarray(var_out.log_likelihood, dtype=float)
        return {
            "scores": ref_ll - var_ll,
            "reference_ll": ref_ll,
            "variant_ll": var_ll,
        }

    def compute_metrics(
        self,
        dataset: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        scores = predictions["scores"]
        labels = dataset["cosmic_label"].to_numpy(dtype=int)
        ref_ll = predictions["reference_ll"]
        var_ll = predictions["variant_ll"]

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        n_total = int(labels.size)
        has_both_classes = n_pos > 0 and n_neg > 0

        metrics: Dict[str, float] = {}

        # ---- Primary: threshold-free ranking metrics ----
        if has_both_classes and n_total >= _MIN_TOTAL_FOR_RANKING_METRICS:
            metrics["auroc"] = float(default_registry.compute("auroc", labels, scores))
            metrics["auprc"] = float(default_registry.compute("auprc", labels, scores))
        else:
            logger.warning(
                "CosmicEvaluator: skipping ranking metrics "
                "(n_pos=%d, n_neg=%d, total=%d < %d)",
                n_pos, n_neg, n_total, _MIN_TOTAL_FOR_RANKING_METRICS,
            )

        # ---- Secondary: threshold-based (fitted on this same set) ----
        threshold_metrics_skipped_reason: Optional[str] = None
        threshold_value: Optional[float] = None
        if n_pos >= _MIN_PER_CLASS_FOR_THRESHOLD_METRICS and n_neg >= _MIN_PER_CLASS_FOR_THRESHOLD_METRICS:
            threshold_value = float(find_optimal_f1_threshold(scores, labels))
            preds = (scores > threshold_value).astype(int)
            metrics["accuracy"] = float(default_registry.compute("accuracy", labels, preds))
            metrics["f1_binary"] = float(default_registry.compute("f1_binary", labels, preds))
            sens, spec = sensitivity_specificity(labels, preds)
            metrics["sensitivity"] = float(sens)
            metrics["specificity"] = float(spec)
            self._last_threshold = threshold_value
            self._last_confusion = confusion_summary(labels, preds)
            self._threshold_fitted_on_this_set = True
        else:
            threshold_metrics_skipped_reason = (
                f"per-class count below {_MIN_PER_CLASS_FOR_THRESHOLD_METRICS} "
                f"(n_pos={n_pos}, n_neg={n_neg})"
            )
            self._last_threshold = None
            self._last_confusion = None
            self._threshold_fitted_on_this_set = False
            logger.warning(
                "CosmicEvaluator: skipping threshold-based metrics (%s)",
                threshold_metrics_skipped_reason,
            )

        # ---- Bootstrap CI ----
        n_boot = int(self.config.get("bootstrap_samples", 200))
        seed = int(self.config.get("seed", 42))
        if has_both_classes and n_total >= _MIN_TOTAL_FOR_RANKING_METRICS:
            self._last_bootstrap = bootstrap_binary_ci(
                labels=labels,
                scores=scores,
                threshold=threshold_value,
                n_boot=n_boot,
                seed=seed,
            )
        else:
            self._last_bootstrap = {}

        # ---- Diagnostics ----
        self._last_score_distribution = score_distribution_stats(
            labels, ref_ll, var_ll, scores
        )
        self._last_threshold_skip_reason = threshold_metrics_skipped_reason
        return metrics

    def build_report(
        self,
        metrics: Dict[str, float],
        dataset: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        report = super().build_report(metrics, dataset, predictions)

        sampling = getattr(self, "_last_sampling", None)
        score_distribution = getattr(self, "_last_score_distribution", None)
        threshold_value = getattr(self, "_last_threshold", None)
        bootstrap_ci = getattr(self, "_last_bootstrap", None) or {}

        bootstrap_payload: Dict[str, Dict[str, float]] = {}
        for key, (lo, hi) in bootstrap_ci.items():
            bootstrap_payload[key] = {"ci_lo": float(lo), "ci_hi": float(hi)}

        # Per-tier composition for the report.
        tier_distribution: Optional[Dict[str, Dict[str, int]]] = None
        if "MUTATION_SIGNIFICANCE_TIER" in dataset.columns:
            tier_series = dataset["MUTATION_SIGNIFICANCE_TIER"].astype(str)
            tier_distribution = (
                dataset.assign(_tier=tier_series)
                .groupby("_tier")["cosmic_label"]
                .value_counts()
                .unstack(fill_value=0)
                .astype(int)
                .rename(columns={0: "neutral", 1: "pathogenic"})
                .to_dict(orient="index")
            )

        preview_count = int(self.config.get("predictions_preview_count", 20))
        prediction_paths = write_predictions(
            output_dir=self.output_dir,
            dataset=dataset,
            predictions=predictions,
            threshold=threshold_value,
            score_distribution=score_distribution,
            sampling=sampling,
            arch=self.handle.arch,
            modality=self.handle.modality,
            preview_count=preview_count,
        )

        report.update(
            {
                "num_examples": int(len(dataset)),
                "class_distribution": {
                    "pathogenic": int((dataset["cosmic_label"] == 1).sum()),
                    "neutral": int((dataset["cosmic_label"] == 0).sum()),
                },
                "sampling": sampling,
                "tier_distribution": tier_distribution,
                "score_distribution": score_distribution,
                "bootstrap_ci_95": bootstrap_payload,
                "threshold": {
                    "optimal_threshold": threshold_value,
                    "fitted_on_this_set": getattr(
                        self, "_threshold_fitted_on_this_set", False
                    ),
                    "skip_reason": getattr(
                        self, "_last_threshold_skip_reason", None
                    ),
                    "confusion_matrix": getattr(self, "_last_confusion", None),
                    "note": (
                        "Threshold-based metrics use the F1-optimal cut "
                        "computed on this same sample (standard zero-shot "
                        "variant-effect convention). Prefer the ranking "
                        "metrics (auroc / auprc) as primary signal; the "
                        "bootstrap CI on each metric is computed on the same "
                        "row resamples at the same threshold so it captures "
                        "sampling variance only."
                    ),
                },
                "artefacts": prediction_paths,
            }
        )
        return report
