"""Replogle Perturb-seq perturbation-response evaluator.

This task falls outside the core decoder/encoder protocol: predicting
expression deltas requires a regression head or a learned adapter on
top of the RNA encoder. The evaluator wires that here:

1. Load the (perturbation, baseline, perturbed) CSV produced by
   :mod:`prepare_csv` (TruthSeq figshare release; see the README of
   ``workflows/data/eval-data-replogle-perturb-seq.sh`` for provenance).
2. Embed each perturbation gene name with the adapter.
3. Fit multi-output Ridge on ``(embedding) -> delta``.
4. Report mean per-perturbation Spearman / Pearson on the held-out
   test split, with bootstrap CIs.

足固め upgrade adds:

- delta-strength-aware subsample (replaces ``df.head(max_examples)``
  so the train+test slice contains both strong and weak KO effects).
- bootstrap 95 % CI on per-perturbation mean Spearman / Pearson
  (resamples whole rows so the CI reflects perturbation-level
  uncertainty, not gene-level).
- per-perturbation predictions log (jsonl + best/worst-fit narrative).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_replogle, stratified_subsample
from .metrics import bootstrap_correlation_ci, delta_correlation
from .predictions_log import write_predictions
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
        self.max_examples: Optional[int] = self.config.get("max_examples")
        self.bootstrap_samples: int = int(self.config.get("bootstrap_samples", 100))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 16)
        )

    def category(self) -> str:
        return "perturbation_response"

    def load_dataset(self) -> pd.DataFrame:
        df = load_replogle(self.replogle_path)
        if self.max_examples is not None and self.max_examples < len(df):
            df = stratified_subsample(
                df,
                n_examples=int(self.max_examples),
                seed=self.seed,
            )
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
        logger.info(
            "Replogle split: train=%d test=%d (test_fraction=%.2f)",
            len(train_df),
            len(test_df),
            self.test_fraction,
        )

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
        return {
            "predictions": preds,
            "observed": test_delta,
            "test_df": test_df,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        observed = predictions["observed"]
        preds = predictions["predictions"]
        metrics = delta_correlation(observed, preds)
        ci = bootstrap_correlation_ci(
            observed,
            preds,
            n_boot=self.bootstrap_samples,
            seed=self.seed,
        )
        self._last_bootstrap_ci = {
            k: {"ci_lo": float(lo), "ci_hi": float(hi)} for k, (lo, hi) in ci.items()
        }
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        artefacts = write_predictions(
            output_dir=self.output_dir,
            test_df=predictions["test_df"],
            observed=predictions["observed"],
            predicted=predictions["predictions"],
            arch=self.handle.arch,
            preview_count=self.predictions_preview_count,
        )
        report.update(
            {
                "num_test_perturbations": int(len(predictions["test_df"])),
                "num_genes": int(predictions["observed"].shape[1])
                if predictions["observed"].size
                else 0,
                "train_size": predictions["train_size"],
                "test_size": predictions["test_size"],
                "seed": self.seed,
                "bootstrap_ci_95": getattr(self, "_last_bootstrap_ci", {}),
                "artefacts": artefacts,
                "notes": (
                    "Adapter embed(perturbation gene name) -> Ridge -> delta. "
                    "Bootstrap CI resamples whole perturbations (rows), so the "
                    "interval reflects perturbation-level uncertainty rather than "
                    "gene-level."
                ),
            }
        )
        return report
