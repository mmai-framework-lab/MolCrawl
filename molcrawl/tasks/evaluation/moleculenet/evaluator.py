"""MoleculeNet evaluator driven by a ModelAdapter.

The evaluator follows the scaffold-split protocol used by MoleculeNet:

1. Load the task CSV and canonicalise SMILES
   (:mod:`.data_preparation`).
2. Build a scaffold split
   (:mod:`.splits`).
3. Embed the train and test subsets through ``ModelAdapter.embed``.
4. Fit a simple linear probe (logistic regression for classification,
   ridge for regression) on the train embeddings.
5. Report the metric pack selected in :mod:`.metrics`.

Adapters that do not support embedding but do support likelihood can
still produce a zero-shot perplexity baseline - the evaluator falls back
to this when embedding is unavailable and reports perplexity plus NaN
placeholders for the task-specific metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import (
    BaseEvaluator,
    ModelHandle,
    default_registry,
)

from .data_preparation import MoleculeNetTaskSpec, get_task, load_dataset
from .metrics import score_classification, score_regression
from .splits import apply_split, random_split, scaffold_split

logger = logging.getLogger(__name__)


class MoleculeNetEvaluator(BaseEvaluator):
    """Evaluate one MoleculeNet task."""

    task_name = "moleculenet"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        task_dir: Path,
        task_spec: MoleculeNetTaskSpec,
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
        self.split_strategy: str = str(self.config.get("split", "scaffold"))
        self.seed: int = int(self.config.get("seed", 0))
        self.val_frac: float = float(self.config.get("val_frac", 0.1))
        self.test_frac: float = float(self.config.get("test_frac", 0.1))

    def category(self) -> str:
        return "property_prediction"

    def load_dataset(self) -> pd.DataFrame:
        df = load_dataset(self.task_dir, self.task_spec)
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            df = df.head(int(max_examples)).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------

    def _build_split(self, df: pd.DataFrame):
        if self.split_strategy == "scaffold":
            return scaffold_split(
                df[self.task_spec.smiles_column].tolist(),
                val_frac=self.val_frac,
                test_frac=self.test_frac,
                seed=self.seed,
            )
        if self.split_strategy == "random":
            return random_split(
                len(df), val_frac=self.val_frac, test_frac=self.test_frac, seed=self.seed
            )
        raise ValueError(f"Unknown split strategy: {self.split_strategy}")

    def run_predictions(self, dataset: pd.DataFrame):
        split = self._build_split(dataset)
        train_df, _val_df, test_df = apply_split(dataset, split)

        adapter = self.adapter
        smi_col = self.task_spec.smiles_column
        label_cols = list(self.task_spec.label_columns)

        if adapter.supports("embedding"):
            train_embed = adapter.embed(train_df[smi_col].tolist()).embeddings
            test_embed = adapter.embed(test_df[smi_col].tolist()).embeddings
            preds = self._probe_predictions(
                train_embed, test_embed, train_df[label_cols], label_cols
            )
            mode = "linear_probe"
        elif adapter.supports("likelihood"):
            ll_out = adapter.score_likelihood(test_df[smi_col].tolist())
            preds = {"log_likelihood": np.asarray(ll_out.log_likelihood, dtype=float)}
            mode = "zero_shot_likelihood"
        else:
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} supports neither embedding nor likelihood; "
                "cannot evaluate MoleculeNet."
            )

        return {
            "mode": mode,
            "test_df": test_df,
            "predictions": preds,
        }

    def _probe_predictions(
        self,
        train_embed: np.ndarray,
        test_embed: np.ndarray,
        train_labels: pd.DataFrame,
        label_cols: List[str],
    ) -> Dict[str, np.ndarray]:
        train_embed = np.asarray(train_embed)
        test_embed = np.asarray(test_embed)

        outputs: Dict[str, np.ndarray] = {}
        if self.task_spec.task_type == "classification":
            from sklearn.linear_model import LogisticRegression

            for col in label_cols:
                y = train_labels[col].to_numpy()
                mask = ~pd.isna(y)
                if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                    outputs[col] = np.full(len(test_embed), np.nan)
                    continue
                clf = LogisticRegression(max_iter=1000)
                clf.fit(train_embed[mask], y[mask].astype(int))
                outputs[col] = clf.predict_proba(test_embed)[:, 1]
            return outputs

        # regression
        from sklearn.linear_model import Ridge

        for col in label_cols:
            y = train_labels[col].to_numpy(dtype=float)
            mask = ~pd.isna(y)
            if mask.sum() < 2:
                outputs[col] = np.full(len(test_embed), np.nan)
                continue
            reg = Ridge(alpha=1.0)
            reg.fit(train_embed[mask], y[mask])
            outputs[col] = reg.predict(test_embed)
        return outputs

    # ------------------------------------------------------------------

    def compute_metrics(self, dataset: pd.DataFrame, predictions):
        mode = predictions["mode"]
        test_df = predictions["test_df"]
        task_spec = self.task_spec

        if mode == "zero_shot_likelihood":
            ll = predictions["predictions"]["log_likelihood"]
            ppl = default_registry.compute("perplexity", float(-np.mean(ll)))
            return {"perplexity": ppl}

        metrics: Dict[str, float] = {}
        preds = predictions["predictions"]
        for col, scores in preds.items():
            y_true = test_df[col].to_numpy()
            mask = ~pd.isna(y_true) & ~pd.isna(scores)
            if mask.sum() < 2:
                logger.warning("Skipping %s: insufficient labelled test rows", col)
                continue
            if task_spec.task_type == "classification":
                sub = score_classification(y_true[mask].astype(int), scores[mask])
            else:
                sub = score_regression(y_true[mask].astype(float), scores[mask])
            for metric_name, value in sub.items():
                metrics[f"{col}.{metric_name}"] = float(value)

        # task-level averages ease the markdown summary
        if metrics:
            for metric_name in set(k.split(".", 1)[1] for k in metrics if "." in k):
                values = [v for k, v in metrics.items() if k.endswith("." + metric_name)]
                metrics[f"mean.{metric_name}"] = float(np.nanmean(values)) if values else float("nan")
        return metrics

    def build_report(self, metrics: Dict[str, float], dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "task_spec": {
                    "name": self.task_spec.name,
                    "task_type": self.task_spec.task_type,
                    "label_columns": list(self.task_spec.label_columns),
                },
                "split_strategy": self.split_strategy,
                "mode": predictions["mode"],
                "num_test_examples": int(len(predictions["test_df"])),
            }
        )
        return report


def evaluate_all(
    handle: ModelHandle,
    base_dir: Path,
    output_dir: Path,
    tasks: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Convenience helper - run the evaluator over a list of task names."""
    from .data_preparation import default_tasks

    specs = default_tasks() if tasks is None else [get_task(name) for name in tasks]
    results: List[Dict[str, Any]] = []
    for spec in specs:
        task_dir = Path(base_dir) / spec.name
        if not task_dir.exists():
            logger.warning("Skipping %s: task directory %s missing", spec.name, task_dir)
            continue
        evaluator = MoleculeNetEvaluator(
            handle=handle,
            output_dir=Path(output_dir) / spec.name,
            task_dir=task_dir,
            task_spec=spec,
            **kwargs,
        )
        results.append(evaluator.run().as_dict())
    return results
