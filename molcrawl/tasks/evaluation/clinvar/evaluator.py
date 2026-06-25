"""Architecture-agnostic ClinVar evaluator.

Uses the model adapter API to score reference and variant sequences,
then reports threshold-free pathogenicity metrics (AUROC / AUPRC /
Spearman) as primary signal, with the F1-optimal threshold-based
metrics retained as a secondary block (flagged as same-set-fitted in
line with the zero-shot variant-effect convention).

The dataset is sampled reproducibly via :func:`sample_clinvar`: by
default the evaluator draws ``n_per_class`` rows from each class with
per-chromosome stratification so the per-chromosome pathogenic-rate
variance (86 % on chrY, 48 % on chrX, 27 % overall) does not leak into
the score. ``n_per_class=None`` runs on the full 221k-variant table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation._base import (
    BaseEvaluator,
    ModelHandle,
    default_registry,
)
from molcrawl.tasks.evaluation import _adapters  # noqa: F401  - registers adapters

from .data_preparation import load_clinvar
from .metrics import (
    confusion_summary,
    find_optimal_f1_threshold,
    score_distribution_stats,
    sensitivity_specificity,
)
from .predictions_log import write_predictions
from .splits import sample_clinvar

logger = logging.getLogger(__name__)

_MIN_PER_CLASS_FOR_THRESHOLD_METRICS = 10
_MIN_TOTAL_FOR_RANKING_METRICS = 20


class ClinVarEvaluator(BaseEvaluator):
    """Binary pathogenicity evaluator for ClinVar variants."""

    task_name = "clinvar"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        clinvar_path: str,
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
        self.clinvar_path = clinvar_path
        self.context_length: int = int(self.config.get("context_length", 512))
        # When set, the adapter only averages PLL over a window of
        # ``± score_window_half`` tokens around the variant centre
        # (default = full sequence). The model still sees full context;
        # only the averaging set is restricted. The centre is taken to
        # be ``flank`` (default 64), matching the upstream window
        # extraction in download_clinvar_sequences().
        self.score_window_half: Optional[int] = self.config.get("score_window_half")
        if self.score_window_half is not None:
            self.score_window_half = int(self.score_window_half)
        self.flank: int = int(self.config.get("flank", 64))

    def category(self) -> str:
        return "variant_effect"

    def load_dataset(self) -> pd.DataFrame:
        df = load_clinvar(self.clinvar_path)

        n_per_class = self.config.get("n_per_class")
        max_examples = self.config.get("max_examples")
        if n_per_class is None and max_examples is not None:
            # Backwards compatibility: legacy callers passed max_examples as a
            # head() slice that produced all-benign samples. Map it to a
            # class-balanced draw of max_examples // 2 per class so old smoke
            # scripts still work but actually yield both classes.
            n_per_class = int(max_examples) // 2 or 1
            logger.warning(
                "ClinVarEvaluator: max_examples=%s is deprecated; re-interpreting "
                "as n_per_class=%d (class-balanced).",
                max_examples,
                n_per_class,
            )

        stratify_chrom = bool(self.config.get("stratify_chrom", True))
        seed = int(self.config.get("seed", 42))

        sampled = sample_clinvar(
            df,
            n_per_class=n_per_class,
            stratify_chrom=stratify_chrom,
            seed=seed,
        )
        # Stash sampling metadata for the report.
        self._last_sampling = {
            "n_per_class": n_per_class,
            "stratify_chrom": stratify_chrom,
            "seed": seed,
            "total_rows_in_file": int(len(df)),
        }
        return sampled

    def run_predictions(self, dataset: pd.DataFrame) -> Dict[str, np.ndarray]:
        adapter = self.adapter
        if not adapter.supports("likelihood"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} does not support likelihood scoring, "
                "which ClinVar requires."
            )

        ref_sequences = dataset["reference_sequence"].astype(str).tolist()
        var_sequences = dataset["variant_sequence"].astype(str).tolist()

        # Build the window-position list once, shared by ref and var
        # scoring so the two PLLs are computed over exactly the same
        # token positions and their difference stays a meaningful score.
        eval_positions: Optional[List[int]] = None
        if self.score_window_half is not None:
            lo = max(0, self.flank - self.score_window_half)
            hi = self.flank + self.score_window_half + 1  # exclusive
            eval_positions = list(range(lo, hi))

        ref_out = adapter.score_likelihood(
            ref_sequences,
            context_length=self.context_length,
            eval_position_indices=eval_positions,
        )
        var_out = adapter.score_likelihood(
            var_sequences,
            context_length=self.context_length,
            eval_position_indices=eval_positions,
        )

        ref_ll = np.asarray(ref_out.log_likelihood, dtype=float)
        var_ll = np.asarray(var_out.log_likelihood, dtype=float)
        # Pathogenicity score: reference is more likely than variant.
        scores = ref_ll - var_ll
        return {
            "scores": scores,
            "reference_log_likelihood": ref_ll,
            "variant_log_likelihood": var_ll,
        }

    def compute_metrics(
        self, dataset: pd.DataFrame, predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        scores = predictions["scores"]
        labels = dataset["pathogenic"].to_numpy(dtype=int)
        ref_ll = predictions["reference_log_likelihood"]
        var_ll = predictions["variant_log_likelihood"]

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        n_total = int(labels.size)
        has_both_classes = n_pos > 0 and n_neg > 0

        metrics: Dict[str, float] = {}

        # ---- Primary: threshold-free ranking metrics ----
        if has_both_classes and n_total >= _MIN_TOTAL_FOR_RANKING_METRICS:
            metrics["auroc"] = float(
                default_registry.compute("auroc", labels, scores)
            )
            metrics["auprc"] = float(
                default_registry.compute("auprc", labels, scores)
            )
            metrics["spearman"] = float(
                default_registry.compute("spearman", labels, scores)
            )
        else:
            logger.warning(
                "ClinVarEvaluator: skipping ranking metrics "
                "(n_pos=%d, n_neg=%d, total=%d < %d)",
                n_pos,
                n_neg,
                n_total,
                _MIN_TOTAL_FOR_RANKING_METRICS,
            )

        # ---- Secondary: threshold-based (fitted on this same set) ----
        threshold_metrics_skipped_reason: Optional[str] = None
        if n_pos >= _MIN_PER_CLASS_FOR_THRESHOLD_METRICS and n_neg >= _MIN_PER_CLASS_FOR_THRESHOLD_METRICS:
            threshold = find_optimal_f1_threshold(scores, labels)
            preds = (scores > threshold).astype(int)
            metrics["accuracy"] = float(
                default_registry.compute("accuracy", labels, preds)
            )
            metrics["f1_binary"] = float(
                default_registry.compute("f1_binary", labels, preds)
            )
            sensitivity, specificity = sensitivity_specificity(labels, preds)
            metrics["sensitivity"] = float(sensitivity)
            metrics["specificity"] = float(specificity)
            self._last_threshold = float(threshold)
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
                "ClinVarEvaluator: skipping threshold-based metrics (%s)",
                threshold_metrics_skipped_reason,
            )

        # ---- Diagnostics: score distribution always computed ----
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

        chrom_distribution = None
        if "chrom" in dataset.columns:
            # The upstream CSV contains mixed-dtype chrom entries (some int,
            # some str); normalise to string before grouping so a single
            # chromosome does not appear twice in the report.
            chrom_series = dataset["chrom"].astype(str)
            chrom_distribution = (
                dataset.assign(_chrom=chrom_series)
                .groupby("_chrom")["pathogenic"]
                .value_counts()
                .unstack(fill_value=0)
                .astype(int)
                .rename(columns={0: "benign", 1: "pathogenic"})
                .to_dict(orient="index")
            )

        sampling = getattr(self, "_last_sampling", None)
        score_distribution = getattr(self, "_last_score_distribution", None)
        threshold_value = getattr(self, "_last_threshold", None)

        # Write per-variant predictions (JSONL) + human-readable narrative.
        # This is always on because the output is cheap (~200 B per row) and
        # it is what makes the evaluator inspectable without re-running.
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
                    "pathogenic": int((dataset["pathogenic"] == 1).sum()),
                    "benign": int((dataset["pathogenic"] == 0).sum()),
                },
                "sampling": sampling,
                "chromosome_distribution": chrom_distribution,
                "score_distribution": score_distribution,
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
                        "Threshold-based metrics use the F1-optimal cut computed "
                        "on this same sample (standard zero-shot variant-effect "
                        "convention). Prefer the ranking metrics (auroc / auprc / "
                        "spearman) as primary signal."
                    ),
                },
                "artefacts": prediction_paths,
            }
        )
        return report
