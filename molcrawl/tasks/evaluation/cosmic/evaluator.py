"""Arch-agnostic COSMIC evaluator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle, default_registry

from .data_preparation import load_cosmic
from .metrics import confusion_summary, find_optimal_f1_threshold, sensitivity_specificity

logger = logging.getLogger(__name__)


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
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            df = df.head(int(max_examples)).reset_index(drop=True)
        return df

    def run_predictions(self, dataset: pd.DataFrame):
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

    def compute_metrics(self, dataset: pd.DataFrame, predictions) -> Dict[str, float]:
        scores = predictions["scores"]
        labels = dataset["cosmic_label"].to_numpy(dtype=int)
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

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "num_examples": int(len(dataset)),
                "optimal_threshold": getattr(self, "_last_threshold", None),
                "confusion_matrix": getattr(self, "_last_confusion", None),
            }
        )
        return report
