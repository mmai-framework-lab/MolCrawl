"""Evaluator for molecule-natural-language pair scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_pairs
from .metrics import summarise

logger = logging.getLogger(__name__)


class MoleculeNatLangEvaluator(BaseEvaluator):
    task_name = "molecule_nat_lang"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        pairs_path: Path,
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
        self.pairs_path = Path(pairs_path)
        self.smiles_column: str = str(self.config.get("smiles_column", "smiles"))
        self.caption_column: str = str(self.config.get("caption_column", "caption"))
        self.template: str = str(self.config.get("template", "{caption}\n{smiles}"))

    def category(self) -> str:
        return "text_alignment"

    def load_dataset(self) -> pd.DataFrame:
        df = load_pairs(self.pairs_path, self.smiles_column, self.caption_column)
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
        formatted = [
            self.template.format(
                caption=row[self.caption_column], smiles=row[self.smiles_column]
            )
            for _, row in dataset.iterrows()
        ]
        out = adapter.score_likelihood(formatted)
        return {"log_likelihoods": np.asarray(out.log_likelihood, dtype=float)}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        return summarise(predictions["log_likelihoods"])

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update({"num_pairs": int(len(dataset)), "template": self.template})
        return report
