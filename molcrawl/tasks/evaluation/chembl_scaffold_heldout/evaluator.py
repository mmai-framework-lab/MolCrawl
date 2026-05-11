"""Held-out evaluator for ChEMBL scaffold splits."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_heldout
from .metrics import perplexity_from_log_likelihoods, probe_metrics

logger = logging.getLogger(__name__)


class ChEMBLScaffoldHeldoutEvaluator(BaseEvaluator):
    """Evaluate compound models on a scaffold held-out ChEMBL split."""

    task_name = "chembl_scaffold_heldout"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        heldout_path: Path,
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
        self.heldout_path = Path(heldout_path)
        self.smiles_column: str = str(self.config.get("smiles_column", "smiles"))
        self.label_column: Optional[str] = self.config.get("label_column")
        self.train_csv: Optional[str] = self.config.get("train_csv")

    def category(self) -> str:
        return "property_prediction" if self.label_column else "generation_quality"

    def load_dataset(self) -> pd.DataFrame:
        df = load_heldout(
            self.heldout_path,
            smiles_column=self.smiles_column,
            label_column=self.label_column,
        )
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            df = df.head(int(max_examples)).reset_index(drop=True)
        return df

    def run_predictions(self, dataset: pd.DataFrame):
        adapter = self.adapter
        smi_list = dataset[self.smiles_column].astype(str).tolist()

        if self.label_column is None:
            if not adapter.supports("likelihood"):
                raise RuntimeError(
                    f"Adapter {type(adapter).__name__} does not support likelihood scoring."
                )
            out = adapter.score_likelihood(smi_list)
            return {"mode": "perplexity", "log_likelihood": np.asarray(out.log_likelihood, dtype=float)}

        if not adapter.supports("embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} does not support embeddings - "
                "cannot fit linear probe."
            )

        if self.train_csv is None:
            raise ValueError(
                "encoder evaluation requires config.train_csv pointing at a scaffold "
                "training split with the same label column."
            )
        train_df = load_heldout(
            Path(self.train_csv),
            smiles_column=self.smiles_column,
            label_column=self.label_column,
        )
        train_embed = np.asarray(
            adapter.embed(train_df[self.smiles_column].astype(str).tolist()).embeddings
        )
        test_embed = np.asarray(adapter.embed(smi_list).embeddings)

        from sklearn.linear_model import LogisticRegression

        y_train = train_df[self.label_column].to_numpy()
        mask = ~pd.isna(y_train)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_embed[mask], y_train[mask].astype(int))
        scores = clf.predict_proba(test_embed)[:, 1]
        return {"mode": "probe", "scores": scores}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        if predictions["mode"] == "perplexity":
            return {"perplexity": perplexity_from_log_likelihoods(predictions["log_likelihood"])}
        scores = predictions["scores"]
        y_true = dataset[self.label_column].to_numpy()
        mask = ~pd.isna(y_true)
        return probe_metrics(y_true[mask].astype(int), scores[mask])

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "num_examples": int(len(dataset)),
                "mode": predictions["mode"],
                "label_column": self.label_column,
            }
        )
        return report
