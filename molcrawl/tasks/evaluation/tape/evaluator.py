"""Evaluator that dispatches to one TAPE sub-task at a time."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import TAPETaskSpec, get_spec, to_frame
from .metrics import (
    classification_metrics,
    contact_prediction_metrics,
    regression_metrics,
)
from .splits import load_splits

logger = logging.getLogger(__name__)


class TAPEEvaluator(BaseEvaluator):
    """Evaluate a protein encoder / decoder on one TAPE task."""

    task_name = "tape"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        task_dir: Path,
        task_spec: TAPETaskSpec,
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
        self.task_dir = Path(task_dir)
        self.task_spec = task_spec

    def category(self) -> str:
        if self.task_spec.task_type == "sequence_labeling":
            return "sequence_annotation"
        if self.task_spec.task_type == "regression":
            return "property_prediction"
        return "property_prediction"

    def load_dataset(self) -> Dict[str, pd.DataFrame]:
        splits = load_splits(self.task_dir, self.task_spec.name)
        frames = {name: to_frame(records, self.task_spec) for name, records in splits.items()}
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            for split in frames:
                frames[split] = frames[split].head(int(max_examples)).reset_index(drop=True)
        return frames

    def run_predictions(self, dataset):
        adapter = self.adapter
        spec = self.task_spec

        if spec.task_type == "sequence_labeling" and spec.name == "contact_prediction":
            # Contact prediction needs residue-residue logits; defer to
            # upstream wiring.  We return a placeholder result so the
            # metric layer can still emit NaN entries.
            return {"mode": "placeholder"}

        if not adapter.supports("embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot produce embeddings; "
                "TAPE evaluation requires encoder embeddings."
            )

        train_df = dataset.get("train")
        test_df = dataset.get("valid")
        if test_df is None:
            test_df = dataset.get("test")
        if train_df is None or test_df is None:
            raise RuntimeError(
                f"TAPE {spec.name}: need at least train + (valid or test) split "
                f"under {self.task_dir}"
            )

        train_emb = np.asarray(
            adapter.embed(train_df[spec.sequence_column].astype(str).tolist()).embeddings
        )
        test_emb = np.asarray(
            adapter.embed(test_df[spec.sequence_column].astype(str).tolist()).embeddings
        )

        y_train = train_df[spec.label_column].to_numpy()
        if spec.task_type == "regression":
            from sklearn.linear_model import Ridge

            reg = Ridge(alpha=1.0)
            reg.fit(train_emb, y_train.astype(float))
            preds = reg.predict(test_emb)
        else:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=1000)
            clf.fit(train_emb, y_train.astype(int))
            preds = clf.predict(test_emb)

        return {"mode": "probe", "predictions": preds, "test_df": test_df}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        spec = self.task_spec
        if predictions.get("mode") == "placeholder":
            return contact_prediction_metrics()
        preds = predictions["predictions"]
        test_df = predictions["test_df"]
        y_true = test_df[spec.label_column].to_numpy()
        if spec.task_type == "regression":
            return regression_metrics(y_true.astype(float), np.asarray(preds, dtype=float))
        return classification_metrics(y_true.astype(int), np.asarray(preds, dtype=int))

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "task_spec": {
                    "name": self.task_spec.name,
                    "task_type": self.task_spec.task_type,
                    "label_column": self.task_spec.label_column,
                },
                "split_sizes": {k: int(len(v)) for k, v in dataset.items()},
            }
        )
        return report


def evaluate_task(handle: ModelHandle, task_name: str, **kwargs: Any):
    spec = get_spec(task_name)
    evaluator = TAPEEvaluator(handle=handle, task_spec=spec, **kwargs)
    return evaluator.run()
