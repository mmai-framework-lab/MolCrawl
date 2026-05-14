"""Held-out evaluator for ChEMBL scaffold splits.

足固め upgrade adds:

- length-stratified subsample (replaces ``df.head(max_examples)`` so
  the held-out evaluation no longer over-represents short / early-id
  ChEMBL records)
- bootstrap 95 % CI on perplexity (and on probe AUROC/AUPRC/acc/F1)
- per-row predictions log (jsonl + narrative TXT showing best-fit /
  worst-fit SMILES under the model)
- length stats on the held-out SMILES so the report makes the
  composition of the test set explicit
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_heldout, stratified_subsample
from .metrics import (
    bootstrap_perplexity_ci,
    bootstrap_probe_ci,
    length_stats,
    perplexity_from_log_likelihoods,
    probe_metrics,
)
from .predictions_log import write_predictions
from .splits import warn_on_scaffold_overlap

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
        self.max_examples: Optional[int] = self.config.get("max_examples")
        self.seed: int = int(self.config.get("seed", 42))
        self.bootstrap_samples: int = int(self.config.get("bootstrap_samples", 100))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 30)
        )

    def category(self) -> str:
        return "property_prediction" if self.label_column else "generation_quality"

    def load_dataset(self) -> pd.DataFrame:
        df = load_heldout(
            self.heldout_path,
            smiles_column=self.smiles_column,
            label_column=self.label_column,
        )
        if self.max_examples is not None and self.max_examples < len(df):
            df = stratified_subsample(
                df,
                n_examples=int(self.max_examples),
                smiles_column=self.smiles_column,
                label_column=self.label_column,
                seed=self.seed,
            )
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
            return {
                "mode": "perplexity",
                "log_likelihood": np.asarray(out.log_likelihood, dtype=float),
                "smiles": smi_list,
            }

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
        # Mirror the held-out cap on the train side so embedding cost stays
        # bounded for very large training CSVs.
        if self.max_examples is not None and self.max_examples < len(train_df):
            train_df = stratified_subsample(
                train_df,
                n_examples=int(self.max_examples),
                smiles_column=self.smiles_column,
                label_column=self.label_column,
                seed=self.seed,
            )

        # Sanity-check the user-provided split is genuinely scaffold-disjoint.
        n_overlap = warn_on_scaffold_overlap(
            train_smiles=train_df[self.smiles_column].astype(str).tolist(),
            test_smiles=smi_list,
        )
        if n_overlap > 0:
            logger.warning(
                "Scaffold leak detected: %d scaffolds appear in both train_csv "
                "and heldout_csv. Held-out perplexity / probe metrics may be "
                "optimistic.",
                n_overlap,
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
        return {
            "mode": "probe",
            "scores": scores,
            "smiles": smi_list,
            "n_train_used": int(mask.sum()),
            "scaffold_overlap": int(n_overlap),
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        if predictions["mode"] == "perplexity":
            ll = predictions["log_likelihood"]
            metrics: Dict[str, float] = {
                "perplexity": perplexity_from_log_likelihoods(ll),
                "mean_log_likelihood": float(np.nanmean(ll)) if len(ll) else float("nan"),
            }
            metrics.update(length_stats(predictions["smiles"]))
            ci_lo, ci_hi = bootstrap_perplexity_ci(
                ll,
                n_boot=self.bootstrap_samples,
                seed=self.seed,
            )
            self._last_bootstrap_ci: Dict[str, Any] = {
                "perplexity": {"ci_lo": ci_lo, "ci_hi": ci_hi}
            }
            return metrics

        scores = predictions["scores"]
        y_true = dataset[self.label_column].to_numpy()
        mask = ~pd.isna(y_true)
        y_true_clean = y_true[mask].astype(int)
        scores_clean = scores[mask]
        metrics = probe_metrics(y_true_clean, scores_clean)
        metrics.update(length_stats(predictions["smiles"]))
        ci = bootstrap_probe_ci(
            y_true_clean,
            scores_clean,
            n_boot=self.bootstrap_samples,
            seed=self.seed,
        )
        self._last_bootstrap_ci = {
            k: {"ci_lo": float(lo), "ci_hi": float(hi)} for k, (lo, hi) in ci.items()
        }
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)

        if predictions["mode"] == "perplexity":
            artefacts = write_predictions(
                output_dir=self.output_dir,
                smiles=predictions["smiles"],
                mode="perplexity",
                log_likelihood=predictions["log_likelihood"],
                arch=self.handle.arch,
                preview_count=self.predictions_preview_count,
            )
        else:
            labels = (
                dataset[self.label_column].to_numpy().tolist()
                if self.label_column
                else None
            )
            artefacts = write_predictions(
                output_dir=self.output_dir,
                smiles=predictions["smiles"],
                mode="probe",
                probe_scores=predictions["scores"],
                labels=labels,
                label_column=self.label_column,
                arch=self.handle.arch,
                preview_count=self.predictions_preview_count,
            )

        report.update(
            {
                "num_examples": int(len(dataset)),
                "mode": predictions["mode"],
                "label_column": self.label_column,
                "seed": self.seed,
                "bootstrap_ci_95": getattr(self, "_last_bootstrap_ci", {}),
                "artefacts": artefacts,
                "notes": (
                    "Scaffold held-out: heldout SMILES come from Bemis-Murcko "
                    "scaffolds disjoint from training. perplexity = exp(-mean LL). "
                    "Lower is better. The predictions narrative samples best- and "
                    "worst-fit SMILES; worst-fit rows are the genuine OOD signal."
                ),
            }
        )
        if predictions["mode"] == "probe":
            report["n_train_used"] = predictions.get("n_train_used")
            report["scaffold_overlap_count"] = predictions.get("scaffold_overlap")
        return report
