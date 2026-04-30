"""Score likelihoods and correlate with gnomAD allele frequency."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_gnomad
from .metrics import correlation_metrics

logger = logging.getLogger(__name__)


class GnomadAFEvaluator(BaseEvaluator):
    task_name = "gnomad_af_correlation"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        gnomad_path: Path,
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
        self.gnomad_path = Path(gnomad_path)
        self.context_length: int = int(self.config.get("context_length", 512))

    def category(self) -> str:
        return "variant_effect"

    def load_dataset(self) -> pd.DataFrame:
        df = load_gnomad(self.gnomad_path)
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
        return {"scores": var_ll - ref_ll}

    def compute_metrics(self, dataset: pd.DataFrame, predictions) -> Dict[str, float]:
        af = dataset["allele_frequency"].to_numpy(dtype=float)
        return correlation_metrics(af, predictions["scores"])

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update({"num_variants": int(len(dataset))})
        return report
