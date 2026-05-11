"""Replogle Perturb-seq perturbation-response evaluator.

This task falls outside the core decoder/encoder protocol: predicting
expression deltas requires a regression head or a learned adapter on
top of the RNA encoder.  The task-centric layout keeps that wiring in
one place: the adapter is expected to expose an ``embedding`` capability,
the evaluator fits a multi-output Ridge on ``(embedding) -> delta``, and
reports mean per-perturbation correlations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_replogle
from .metrics import delta_correlation
from .splits import perturbation_split

logger = logging.getLogger(__name__)


class ReploglePerturbSeqEvaluator(BaseEvaluator):
    task_name = "replogle_perturb_seq"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        replogle_path: Path,
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
        self.replogle_path = Path(replogle_path)
        self.test_fraction: float = float(self.config.get("test_fraction", 0.2))
        self.seed: int = int(self.config.get("seed", 0))

    def category(self) -> str:
        return "perturbation_response"

    def load_dataset(self) -> pd.DataFrame:
        df = load_replogle(self.replogle_path)
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            df = df.head(int(max_examples)).reset_index(drop=True)
        return df

    def _delta_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [np.asarray(p) - np.asarray(b) for p, b in zip(df["perturbed"], df["baseline"])],
            dtype=float,
        )

    def run_predictions(self, dataset: pd.DataFrame):
        adapter = self.adapter
        if not adapter.supports("embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot embed; Perturb-seq requires embeddings."
            )
        train_df, test_df = perturbation_split(
            dataset, test_fraction=self.test_fraction, seed=self.seed
        )
        if test_df.empty:
            raise RuntimeError("Replogle split produced an empty test set")

        train_emb = np.asarray(
            adapter.embed(train_df["perturbation"].astype(str).tolist()).embeddings
        )
        test_emb = np.asarray(
            adapter.embed(test_df["perturbation"].astype(str).tolist()).embeddings
        )
        train_delta = self._delta_matrix(train_df)
        test_delta = self._delta_matrix(test_df)

        from sklearn.linear_model import Ridge

        reg = Ridge(alpha=1.0)
        reg.fit(train_emb, train_delta)
        preds = reg.predict(test_emb)
        return {"predictions": preds, "observed": test_delta, "test_df": test_df}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        return delta_correlation(predictions["observed"], predictions["predictions"])

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "num_test_perturbations": int(len(predictions["test_df"])),
                "num_genes": int(predictions["observed"].shape[1])
                if predictions["observed"].size else 0,
            }
        )
        return report
