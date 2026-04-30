"""Tabula Sapiens cell-type classification evaluator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_jsonl
from .metrics import cell_type_metrics
from .splits import cross_tissue_split, random_split

logger = logging.getLogger(__name__)


def _tokens_to_strings(tokens: Sequence[Sequence[int]]) -> List[str]:
    return [" ".join(str(int(t)) for t in cell) for cell in tokens]


class TabulaSapiensEvaluator(BaseEvaluator):
    task_name = "tabula_sapiens"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        jsonl_path: Path,
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
        self.jsonl_path = Path(jsonl_path)
        self.test_fraction: float = float(self.config.get("test_fraction", 0.2))
        self.seed: int = int(self.config.get("seed", 0))
        self.holdout_tissues: Optional[List[str]] = self.config.get("holdout_tissues")

    def category(self) -> str:
        return "cell_type_annotation"

    def load_dataset(self):
        return load_jsonl(self.jsonl_path, max_cells=self.config.get("max_cells"))

    def run_predictions(self, dataset):
        adapter = self.adapter
        if not adapter.supports("embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot embed; cell-type annotation requires embeddings."
            )

        strings = _tokens_to_strings(dataset["tokens"])
        labels = np.asarray(dataset["cell_type"])

        if self.holdout_tissues is not None:
            train_idx, test_idx = cross_tissue_split(dataset["tissue"], self.holdout_tissues)
        else:
            train_idx, test_idx = random_split(
                len(strings), test_fraction=self.test_fraction, seed=self.seed
            )
        if len(test_idx) == 0:
            raise RuntimeError("TabulaSapiens split produced an empty test set")

        train_strings = [strings[i] for i in train_idx]
        test_strings = [strings[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        train_emb = np.asarray(adapter.embed(train_strings).embeddings)
        test_emb = np.asarray(adapter.embed(test_strings).embeddings)

        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_emb, train_labels)
        preds = clf.predict(test_emb)
        return {"predictions": preds, "test_labels": test_labels, "test_size": int(len(test_idx))}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        return cell_type_metrics(
            np.asarray(predictions["test_labels"]),
            np.asarray(predictions["predictions"]),
        )

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "test_size": predictions["test_size"],
                "num_classes": int(len(set(dataset["cell_type"]))),
            }
        )
        return report
