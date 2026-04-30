"""Architecture-agnostic ClinVar evaluator.

Uses the model adapter API to score reference and variant sequences,
then re-uses the existing pathogenicity-score protocol (likelihood
difference between reference and variant, with an F1-optimal threshold).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation._base import (
    BaseEvaluator,
    ModelHandle,
    default_registry,
)
from molcrawl.tasks.evaluation import _adapters  # noqa: F401  - registers adapters

from .data_preparation import load_clinvar
from .metrics import confusion_summary, find_optimal_f1_threshold, sensitivity_specificity

logger = logging.getLogger(__name__)


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

    def category(self) -> str:
        return "variant_effect"

    def load_dataset(self) -> pd.DataFrame:
        df = load_clinvar(self.clinvar_path)
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            df = df.head(int(max_examples)).reset_index(drop=True)
        return df

    def run_predictions(self, dataset: pd.DataFrame) -> Dict[str, np.ndarray]:
        adapter = self.adapter
        if not adapter.supports("likelihood"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} does not support likelihood scoring, "
                "which ClinVar requires."
            )

        ref_sequences = dataset["reference_sequence"].astype(str).tolist()
        var_sequences = dataset["variant_sequence"].astype(str).tolist()

        ref_out = adapter.score_likelihood(
            ref_sequences, context_length=self.context_length
        )
        var_out = adapter.score_likelihood(
            var_sequences, context_length=self.context_length
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
        threshold = find_optimal_f1_threshold(scores, labels)
        preds = (scores > threshold).astype(int)

        metrics: Dict[str, float] = {
            "accuracy": default_registry.compute("accuracy", labels, preds),
            "f1_binary": default_registry.compute("f1_binary", labels, preds),
        }
        if len(np.unique(labels)) >= 2:
            metrics["auroc"] = default_registry.compute("auroc", labels, scores)
            metrics["auprc"] = default_registry.compute("auprc", labels, scores)

        sensitivity, specificity = sensitivity_specificity(labels, preds)
        metrics["sensitivity"] = sensitivity
        metrics["specificity"] = specificity

        self._last_threshold = float(threshold)
        self._last_confusion = confusion_summary(labels, preds)
        return metrics

    def build_report(
        self,
        metrics: Dict[str, float],
        dataset: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "num_examples": int(len(dataset)),
                "class_distribution": {
                    "pathogenic": int((dataset["pathogenic"] == 1).sum()),
                    "benign": int((dataset["pathogenic"] == 0).sum()),
                },
                "optimal_threshold": getattr(self, "_last_threshold", None),
                "confusion_matrix": getattr(self, "_last_confusion", None),
            }
        )
        return report
