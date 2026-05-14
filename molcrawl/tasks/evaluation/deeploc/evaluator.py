"""DeepLoc 2.0 subcellular localisation evaluator.

足固め upgrade adds:

- class-balanced subsample (replaces ``df.head(max_examples)`` so
  rare classes — Peroxisome, Lysosome — survive into the test split)
- bootstrap 95 % CI on accuracy / f1_macro / mcc
- per-protein predictions log (jsonl + per-kingdom narrative TXT)
- test-set composition summary (cluster ids used for test, per-class
  count, per-kingdom count) surfaced in the report
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import DEEPLOC_CLASSES, load_deeploc, stratified_subsample
from .metrics import bootstrap_multiclass_ci, multiclass_metrics
from .predictions_log import write_predictions
from .splits import cluster_split

logger = logging.getLogger(__name__)


class DeepLocEvaluator(BaseEvaluator):
    """Encoder-probe evaluator for DeepLoc 2.0."""

    task_name = "deeploc"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        deeploc_path: Path,
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
        self.deeploc_path = Path(deeploc_path)
        self.test_fraction: float = float(self.config.get("test_fraction", 0.2))
        self.seed: int = int(self.config.get("seed", 0))
        self.max_examples: Optional[int] = self.config.get("max_examples")
        self.bootstrap_samples: int = int(self.config.get("bootstrap_samples", 100))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 20)
        )

    def category(self) -> str:
        return "property_prediction"

    def load_dataset(self) -> pd.DataFrame:
        df = load_deeploc(self.deeploc_path)
        df["label"] = df["localisation"].map(_label_to_index)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        if self.max_examples is not None and self.max_examples < len(df):
            df = stratified_subsample(
                df,
                n_examples=int(self.max_examples),
                label_column="localisation",
                seed=self.seed,
            )
        return df

    def run_predictions(self, dataset: pd.DataFrame):
        adapter = self.adapter
        if not adapter.supports("embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot embed; DeepLoc requires encoder embeddings."
            )
        train_df, test_df = cluster_split(
            dataset, test_fraction=self.test_fraction, seed=self.seed
        )
        logger.info(
            "DeepLoc split: train=%d test=%d (test_fraction=%.2f)",
            len(train_df),
            len(test_df),
            self.test_fraction,
        )
        train_emb = np.asarray(
            adapter.embed(train_df["sequence"].astype(str).tolist()).embeddings
        )
        test_emb = np.asarray(
            adapter.embed(test_df["sequence"].astype(str).tolist()).embeddings
        )
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_emb, train_df["label"].astype(int).to_numpy())
        preds = clf.predict(test_emb)
        return {
            "predictions": preds,
            "test_df": test_df,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        test_df = predictions["test_df"]
        y_true = test_df["label"].astype(int).to_numpy()
        y_pred = np.asarray(predictions["predictions"], dtype=int)
        metrics = multiclass_metrics(y_true, y_pred)
        ci = bootstrap_multiclass_ci(
            y_true,
            y_pred,
            n_boot=self.bootstrap_samples,
            seed=self.seed,
        )
        self._last_bootstrap_ci = {
            k: {"ci_lo": float(lo), "ci_hi": float(hi)} for k, (lo, hi) in ci.items()
        }
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        test_df = predictions["test_df"]

        artefacts = write_predictions(
            output_dir=self.output_dir,
            test_df=test_df,
            y_pred=predictions["predictions"],
            classes=DEEPLOC_CLASSES,
            arch=self.handle.arch,
            preview_count=self.predictions_preview_count,
        )

        # Test-set composition for the report
        composition: Dict[str, Any] = {}
        if "kingdom" in test_df.columns:
            composition["per_kingdom"] = (
                test_df["kingdom"].astype(str).value_counts().to_dict()
            )
        composition["per_class"] = (
            test_df["localisation"].astype(str).value_counts().to_dict()
        )
        if "cluster_id" in test_df.columns:
            composition["test_clusters"] = sorted(
                int(c) for c in test_df["cluster_id"].astype(int).unique()
            )

        report.update(
            {
                "num_classes": len(DEEPLOC_CLASSES),
                "train_size": predictions["train_size"],
                "test_size": predictions["test_size"],
                "seed": self.seed,
                "bootstrap_ci_95": getattr(self, "_last_bootstrap_ci", {}),
                "test_composition": composition,
                "artefacts": artefacts,
                "notes": (
                    "Encoder-probe over a logistic-regression head. Train / test "
                    "split honours DeepLoc Partition (cluster id) so homologous "
                    "proteins do not leak across splits. Multi-localised proteins "
                    "are collapsed to their dominant compartment in prepare_csv."
                ),
            }
        )
        return report


def _label_to_index(name: object) -> float:
    text = str(name).strip()
    try:
        return float(DEEPLOC_CLASSES.index(text))
    except ValueError:
        return float("nan")
