"""GUE encoder-probe evaluator.

For each sub-task: embed train sequences, fit a logistic-regression
probe, score the test split (or dev when test is missing).

足固め upgrade adds:

- per-split class-balanced subsample (replaces df.head(max_examples)
  so rare classes — especially in covid_variants and splice — survive
  into both train and test)
- bootstrap 95 % CI on accuracy / f1_macro / mcc / f1_binary
- per-row predictions log (jsonl + per-class confusion narrative TXT)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import (
    GUETaskSpec,
    get_spec,
    load_splits,
    stratified_subsample,
)
from .metrics import bootstrap_classification_ci, classification_metrics
from .predictions_log import write_predictions

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
        self.max_examples: Optional[int] = self.config.get("max_examples")
        self.seed: int = int(self.config.get("seed", 42))
        self.bootstrap_samples: int = int(self.config.get("bootstrap_samples", 100))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 16)
        )

    def category(self) -> str:
        return "sequence_annotation"

    def load_dataset(self):
        frames = load_splits(self.task_dir)
        if self.max_examples is not None:
            for split in list(frames.keys()):
                frames[split] = stratified_subsample(
                    frames[split],
                    n_examples=int(self.max_examples),
                    label_column=self.task_spec.label_column,
                    seed=self.seed,
                )
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

        logger.info(
            "GUE %s: embedding train=%d test=%d (num_classes=%d)",
            self.task_spec.name,
            len(train_df),
            len(test_df),
            self.task_spec.num_classes,
        )
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
        return {
            "predictions": preds,
            "test_df": test_df,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        test_df = predictions["test_df"]
        y_true = test_df[self.task_spec.label_column].astype(int).to_numpy()
        y_pred = np.asarray(predictions["predictions"], dtype=int)
        metrics = classification_metrics(y_true, y_pred, self.task_spec.num_classes)
        ci = bootstrap_classification_ci(
            y_true,
            y_pred,
            num_classes=self.task_spec.num_classes,
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
            y_pred=predictions["predictions"],
            task_name=self.task_spec.name,
            num_classes=self.task_spec.num_classes,
            sequence_column=self.task_spec.sequence_column,
            label_column=self.task_spec.label_column,
            arch=self.handle.arch,
            preview_count=self.predictions_preview_count,
        )
        report.update(
            {
                "task_spec": {
                    "name": self.task_spec.name,
                    "num_classes": self.task_spec.num_classes,
                },
                "split_sizes": {k: int(len(v)) for k, v in dataset.items()},
                "train_size_used": predictions["train_size"],
                "test_size_used": predictions["test_size"],
                "seed": self.seed,
                "bootstrap_ci_95": getattr(self, "_last_bootstrap_ci", {}),
                "artefacts": artefacts,
                "notes": (
                    "Encoder-probe over a logistic-regression head. "
                    "Multi-class tasks (splice_reconstructed=3, covid_variants=9) "
                    "report accuracy/f1_macro; binary tasks add mcc/f1_binary."
                ),
            }
        )
        return report


def evaluate_task(handle: ModelHandle, task_name: str, **kwargs: Any):
    spec = get_spec(task_name)
    evaluator = GUEEvaluator(handle=handle, task_spec=spec, **kwargs)
    return evaluator.run()
