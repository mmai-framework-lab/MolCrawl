"""GUE encoder-probe evaluator.

For each sub-task embed the training sequences, fit a logistic
regression probe, and score the test split.  Uses ``dev`` as a fallback
when ``test`` is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import GUETaskSpec, get_spec, load_splits
from .metrics import classification_metrics

logger = logging.getLogger(__name__)


class GUEEvaluator(BaseEvaluator):
    task_name = "gue"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        task_dir: Path,
        task_spec: GUETaskSpec,
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
        return "sequence_annotation"

    def load_dataset(self):
        frames = load_splits(self.task_dir)
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            for split in frames:
                frames[split] = frames[split].head(int(max_examples)).reset_index(drop=True)
        return frames

    def run_predictions(self, dataset):
        adapter = self.adapter
        if not adapter.supports("embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot embed; GUE requires encoder embeddings."
            )
        train_df = dataset["train"]
        test_df = dataset.get("test")
        if test_df is None:
            test_df = dataset.get("dev")
        if test_df is None:
            raise RuntimeError("GUE needs a test or dev split")

        train_emb = np.asarray(
            adapter.embed(train_df[self.task_spec.sequence_column].astype(str).tolist()).embeddings
        )
        test_emb = np.asarray(
            adapter.embed(test_df[self.task_spec.sequence_column].astype(str).tolist()).embeddings
        )
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_emb, train_df[self.task_spec.label_column].astype(int).to_numpy())
        preds = clf.predict(test_emb)
        return {"predictions": preds, "test_df": test_df}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        test_df = predictions["test_df"]
        y_true = test_df[self.task_spec.label_column].astype(int).to_numpy()
        y_pred = np.asarray(predictions["predictions"], dtype=int)
        return classification_metrics(y_true, y_pred, self.task_spec.num_classes)

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "task_spec": {
                    "name": self.task_spec.name,
                    "num_classes": self.task_spec.num_classes,
                },
                "split_sizes": {k: int(len(v)) for k, v in dataset.items()},
            }
        )
        return report


def evaluate_task(handle: ModelHandle, task_name: str, **kwargs: Any):
    spec = get_spec(task_name)
    evaluator = GUEEvaluator(handle=handle, task_spec=spec, **kwargs)
    return evaluator.run()
