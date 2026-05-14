"""MoleculeNet evaluator driven by a ModelAdapter.

The evaluator follows the scaffold-split protocol used by MoleculeNet:

1. Load the task CSV and canonicalise SMILES
   (:mod:`.data_preparation`).
2. Optionally stratify-subsample before splitting so smoke runs still
   exercise both classes / all quantiles (:func:`.splits.stratified_subsample`).
3. Build a scaffold split (:mod:`.splits`).
4. Embed the train and test subsets through ``ModelAdapter.embed``.
5. Fit a simple linear probe (logistic regression for classification,
   ridge for regression) on the train embeddings.
6. Report the metric pack selected in :mod:`.metrics`, add bootstrap
   95 % CIs for the ranking / regression primaries, and emit
   ``predictions.jsonl`` + ``predictions.txt`` so the user can
   inspect what the probe actually did without re-running.

Adapters that do not support embedding but do support likelihood can
still produce a zero-shot perplexity baseline — the evaluator falls
back to this when embedding is unavailable and flags the mode in the
report.
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
from .metrics import (
    bootstrap_ci,
    score_classification,
    score_regression,
    split_label_distribution,
)
from .predictions_log import write_predictions
from .splits import apply_split, random_split, scaffold_split, stratified_subsample

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

        n_examples = self.config.get("n_examples")
        max_examples = self.config.get("max_examples")
        if n_examples is None and max_examples is not None:
            n_examples = int(max_examples)
            logger.warning(
                "MoleculeNetEvaluator: max_examples=%s deprecated; "
                "re-interpreting as n_examples=%d with stratified subsample.",
                max_examples,
                n_examples,
            )

        if n_examples is not None and n_examples < len(df):
            df = stratified_subsample(
                df,
                n_examples=int(n_examples),
                label_columns=list(self.task_spec.label_columns),
                task_type=self.task_spec.task_type,
                seed=int(self.config.get("subsample_seed", self.seed + 1)),
            )

        self._last_sampling = {
            "n_examples": n_examples,
            "split": self.split_strategy,
            "val_frac": self.val_frac,
            "test_frac": self.test_frac,
            "seed": self.seed,
            "total_after_subsample": int(len(df)),
        }
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
        train_df, val_df, test_df = apply_split(dataset, split)

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

        self._last_split_sizes = {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        }
        self._last_label_distribution = {
            split_name: {
                col: split_label_distribution(
                    split_df[col].to_numpy(), self.task_spec.task_type
                )
                for col in label_cols
            }
            for split_name, split_df in (
                ("train", train_df),
                ("val", val_df),
                ("test", test_df),
            )
        }

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
            self._last_bootstrap = {}
            return {"perplexity": float(ppl)}

        metrics: Dict[str, float] = {}
        bootstrap_payload: Dict[str, Dict[str, Dict[str, float]]] = {}
        preds = predictions["predictions"]
        n_boot = int(self.config.get("bootstrap_samples", 200))
        seed = int(self.config.get("seed", 0))

        for col, scores in preds.items():
            y_true = test_df[col].to_numpy()
            mask = ~pd.isna(y_true) & ~pd.isna(scores)
            if mask.sum() < 2:
                logger.warning("Skipping %s: insufficient labelled test rows", col)
                continue
            if task_spec.task_type == "classification":
                sub = score_classification(
                    y_true[mask].astype(int), np.asarray(scores)[mask]
                )
                ci = bootstrap_ci(
                    y_true[mask].astype(int),
                    np.asarray(scores)[mask],
                    task_type="classification",
                    n_boot=n_boot,
                    seed=seed,
                )
            else:
                sub = score_regression(
                    y_true[mask].astype(float), np.asarray(scores)[mask]
                )
                ci = bootstrap_ci(
                    y_true[mask].astype(float),
                    np.asarray(scores)[mask],
                    task_type="regression",
                    n_boot=n_boot,
                    seed=seed,
                )
            for metric_name, value in sub.items():
                metrics[f"{col}.{metric_name}"] = float(value)
            if ci:
                bootstrap_payload[col] = {
                    key: {"ci_lo": float(lo), "ci_hi": float(hi)}
                    for key, (lo, hi) in ci.items()
                }

        # task-level averages ease the markdown summary
        if metrics:
            metric_names = set(k.split(".", 1)[1] for k in metrics if "." in k)
            for metric_name in metric_names:
                values = [v for k, v in metrics.items() if k.endswith("." + metric_name)]
                metrics[f"mean.{metric_name}"] = (
                    float(np.nanmean(values)) if values else float("nan")
                )

        self._last_bootstrap = bootstrap_payload
        return metrics

    def build_report(self, metrics: Dict[str, float], dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)

        preview_count = int(self.config.get("predictions_preview_count", 20))
        prediction_paths = write_predictions(
            output_dir=self.output_dir,
            test_df=predictions["test_df"],
            preds=predictions["predictions"],
            label_columns=list(self.task_spec.label_columns),
            smiles_column=self.task_spec.smiles_column,
            task_type=self.task_spec.task_type,
            mode=predictions["mode"],
            split_sizes=getattr(self, "_last_split_sizes", None),
            arch=self.handle.arch,
            preview_count=preview_count,
        )

        report.update(
            {
                "task_spec": {
                    "name": self.task_spec.name,
                    "task_type": self.task_spec.task_type,
                    "label_columns": list(self.task_spec.label_columns),
                },
                "split_strategy": self.split_strategy,
                "split_sizes": getattr(self, "_last_split_sizes", None),
                "label_distribution": getattr(
                    self, "_last_label_distribution", None
                ),
                "mode": predictions["mode"],
                "num_test_examples": int(len(predictions["test_df"])),
                "sampling": getattr(self, "_last_sampling", None),
                "bootstrap_ci_95": getattr(self, "_last_bootstrap", None),
                "artefacts": prediction_paths,
                "notes": (
                    "Linear probe (LogReg / Ridge) on top of ModelAdapter.embed. "
                    "Classification primaries: AUROC / AUPRC. Regression "
                    "primaries: RMSE / R² / Spearman. Bootstrap 95 % CIs are "
                    "computed per label column under the same seed."
                ),
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
