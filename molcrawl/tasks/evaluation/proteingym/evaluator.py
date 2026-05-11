"""Arch-agnostic ProteinGym evaluator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_proteingym
from .metrics import correlation_metrics, optional_binary_metrics

logger = logging.getLogger(__name__)


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
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            df = df.head(int(max_examples)).reset_index(drop=True)
        return df

    def run_predictions(self, dataset: pd.DataFrame):
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
        self, dataset: pd.DataFrame, predictions
    ) -> Dict[str, float]:
        scores = predictions["scores"]
        dms = dataset["DMS_score"].to_numpy(dtype=float)
        metrics = correlation_metrics(dms, scores)
        if "DMS_bin_score" in dataset.columns:
            y_bin = dataset["DMS_bin_score"].to_numpy()
            extra = optional_binary_metrics(y_bin.astype(int), scores)
            metrics.update(extra)
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "num_variants": int(len(dataset)),
                "context_length": self.context_length,
            }
        )
        return report
