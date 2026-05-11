"""DeepLoc 2.0 subcellular localisation evaluator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import DEEPLOC_CLASSES, load_deeploc
from .metrics import multiclass_metrics
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

    def category(self) -> str:
        return "property_prediction"

    def load_dataset(self) -> pd.DataFrame:
        df = load_deeploc(self.deeploc_path)
        # Normalise labels to contiguous integers.
        df["label"] = df["localisation"].map(_label_to_index)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            df = df.head(int(max_examples)).reset_index(drop=True)
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
        return {"predictions": preds, "test_df": test_df}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        test_df = predictions["test_df"]
        y_true = test_df["label"].astype(int).to_numpy()
        y_pred = np.asarray(predictions["predictions"], dtype=int)
        return multiclass_metrics(y_true, y_pred)

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "num_classes": len(DEEPLOC_CLASSES),
                "test_size": int(len(predictions["test_df"])),
            }
        )
        return report


def _label_to_index(name: object) -> float:
    text = str(name).strip()
    try:
        return float(DEEPLOC_CLASSES.index(text))
    except ValueError:
        return float("nan")
