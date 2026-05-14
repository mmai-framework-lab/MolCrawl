"""Evaluator for molecule-natural-language pair scoring.

足固め upgrade adds:

- combined-length-stratified subsample (replaces df.head(max_examples))
- bootstrap 95 % CI on perplexity
- per-pair predictions log (jsonl + best/worst-fit narrative TXT)
- length stats so the report makes the scored corpus shape explicit
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_pairs, stratified_subsample
from .metrics import bootstrap_perplexity_ci, summarise
from .predictions_log import write_predictions

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
        self.max_examples: Optional[int] = self.config.get("max_examples")
        self.seed: int = int(self.config.get("seed", 42))
        self.bootstrap_samples: int = int(self.config.get("bootstrap_samples", 100))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 20)
        )

    def category(self) -> str:
        return "text_alignment"

    def load_dataset(self) -> pd.DataFrame:
        df = load_pairs(self.pairs_path, self.smiles_column, self.caption_column)
        if self.max_examples is not None and self.max_examples < len(df):
            df = stratified_subsample(
                df,
                n_examples=int(self.max_examples),
                smiles_column=self.smiles_column,
                caption_column=self.caption_column,
                seed=self.seed,
            )
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
        return {
            "log_likelihoods": np.asarray(out.log_likelihood, dtype=float),
            "formatted_lengths": [len(s) for s in formatted],
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        ll = predictions["log_likelihoods"]
        metrics: Dict[str, float] = dict(summarise(ll))
        ci_lo, ci_hi = bootstrap_perplexity_ci(
            ll, n_boot=self.bootstrap_samples, seed=self.seed
        )
        self._last_bootstrap_ci = {"perplexity": {"ci_lo": ci_lo, "ci_hi": ci_hi}}
        if predictions.get("formatted_lengths"):
            lengths = np.asarray(predictions["formatted_lengths"], dtype=float)
            metrics.update(
                {
                    "formatted_length_mean": float(lengths.mean()),
                    "formatted_length_median": float(np.median(lengths)),
                    "formatted_length_max": float(lengths.max()),
                }
            )
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        artefacts = write_predictions(
            output_dir=self.output_dir,
            pairs=dataset,
            log_likelihoods=predictions["log_likelihoods"],
            smiles_column=self.smiles_column,
            caption_column=self.caption_column,
            template=self.template,
            arch=self.handle.arch,
            preview_count=self.predictions_preview_count,
        )
        report.update(
            {
                "num_pairs": int(len(dataset)),
                "template": self.template,
                "seed": self.seed,
                "bootstrap_ci_95": getattr(self, "_last_bootstrap_ci", {}),
                "artefacts": artefacts,
                "notes": (
                    "Pseudo / causal perplexity of formatted "
                    "(caption + molecule) pairs. Lower is better. "
                    "Best-fit examples are pairs the model reproduces fluently; "
                    "worst-fit examples are corpus-mismatched (long SELFIES, "
                    "terse captions, or rare entities)."
                ),
            }
        )
        return report
