"""Tabula Sapiens cell-type classification evaluator.

consolidation upgrade adds:

- class-balanced subsample over ``cell_type`` (replaces unbounded
  ``max_cells`` head-clipping).
- bootstrap 95 % CI on accuracy / f1_macro.
- per-cell predictions log (jsonl + per-class CORRECT vs WRONG
  narrative TXT).
- token ids passed through to the adapter directly (no string
  round-trip), matching the rna_benchmark fix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_jsonl, stratified_subsample
from .metrics import bootstrap_celltype_ci, cell_type_metrics, drop_rare_labels
from .predictions_log import write_predictions
from .splits import cross_tissue_split, precomputed_split, random_split

logger = logging.getLogger(__name__)


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
        self.max_cells: Optional[int] = self.config.get("max_cells")
        self.bootstrap_samples: int = int(self.config.get("bootstrap_samples", 100))
        # A cell may be longer than the adapter's default window. RNA packs the
        # top 1,024 ranked genes into one line, and embedding at 512 would drop
        # half of every cell without saying so.
        self.context_length: int = int(self.config.get("context_length", 1024))
        self.embed_batch_size: int = int(self.config.get("embed_batch_size", 16))
        # Classes below this in the *test* side are set aside before the macro
        # average, and named in the result.
        self.min_cells_per_label: int = int(self.config.get("min_cells_per_label", 20))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 16)
        )

    def category(self) -> str:
        return "cell_type_annotation"

    def load_dataset(self):
        # Load everything (max_cells is a sampling cap, not a head cap), then
        # stratify by class.
        ds = load_jsonl(self.jsonl_path, max_cells=None)
        if self.max_cells is not None and self.max_cells < len(ds["tokens"]):
            # Stratifying by class runs across both sides of a precomputed
            # split, so the train/test proportions move and a class that is
            # thin on the test side can lose it entirely. Fine for a timing
            # run, not for a reported number.
            if ds.get("split") and any(str(x) for x in ds["split"]):
                logger.warning(
                    "max_cells=%s subsamples across the split the JSONL carries; "
                    "the reported proportions will not be the split's own.",
                    self.max_cells,
                )
            ds = stratified_subsample(
                ds, n_examples=int(self.max_cells), seed=self.seed
            )
        return ds

    def run_predictions(self, dataset):
        adapter = self.adapter
        if not adapter.supports("embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot embed; cell-type annotation requires embeddings."
            )

        tokens = dataset["tokens"]
        labels = np.asarray(dataset["cell_type"])

        # A split the JSONL already carries wins over one drawn here. Splitting
        # cells at random puts the same donor on both sides, and a linear probe
        # then has donor-specific signal available instead of cell type -- which
        # is the thing being measured.
        split_labels = dataset.get("split")
        if split_labels and any(str(x) for x in split_labels):
            train_idx, test_idx = precomputed_split(split_labels)
            self._split_source = "from the JSONL"
            logger.info(
                "TabulaSapiens split: taken from the JSONL (train=%d test=%d)",
                len(train_idx), len(test_idx),
            )
        elif self.holdout_tissues is not None:
            train_idx, test_idx = cross_tissue_split(dataset["tissue"], self.holdout_tissues)
            self._split_source = f"holdout tissues {self.holdout_tissues}"
        else:
            train_idx, test_idx = random_split(
                len(tokens), test_fraction=self.test_fraction, seed=self.seed
            )
            self._split_source = f"random, test_fraction={self.test_fraction:.2f}"
        if len(test_idx) == 0:
            raise RuntimeError("TabulaSapiens split produced an empty test set")

        train_tokens = [tokens[i] for i in train_idx]
        test_tokens = [tokens[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        logger.info(
            "TabulaSapiens split: train=%d test=%d (%s)",
            len(train_idx),
            len(test_idx),
            # Naming test_fraction here regardless was misleading: it is not what
            # produced the split when the JSONL carried one.
            self._split_source,
        )
        # Pass token-id lists straight to the adapter (HfMlm.embed accepts
        # both strings and pre-tokenised int lists).
        _kw = {"context_length": self.context_length, "batch_size": self.embed_batch_size}
        logger.info("Embedding with context_length=%d", self.context_length)
        train_emb = np.asarray(adapter.embed(train_tokens, **_kw).embeddings)
        test_emb = np.asarray(adapter.embed(test_tokens, **_kw).embeddings)

        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_emb, train_labels)
        preds = clf.predict(test_emb)
        return {
            "predictions": preds,
            "test_labels": test_labels,
            "test_tokens": test_tokens,
            "test_tissue": [dataset["tissue"][i] for i in test_idx],
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        yt_all = np.asarray(predictions["test_labels"])
        yp_all = np.asarray(predictions["predictions"])
        # f1_macro weights every class equally, so a class with a handful of
        # cells in the test side swings the average on one or two predictions.
        # Set those aside, and name them -- a smaller problem reported as the
        # whole one is worse than the noise it removes.
        yt, yp, dropped = drop_rare_labels(yt_all, yp_all, self.min_cells_per_label)
        self._dropped_labels = dropped
        if dropped:
            logger.info(
                "Excluded %d classes under %d cells in the test side: %s",
                len(dropped), self.min_cells_per_label,
                ", ".join(f"{k} ({v})" for k, v in sorted(dropped.items())),
            )
        metrics = cell_type_metrics(yt, yp)
        metrics["n_classes_scored"] = int(len(np.unique(yt)))
        metrics["n_classes_excluded"] = len(dropped)
        metrics["n_cells_scored"] = int(len(yt))
        ci = bootstrap_celltype_ci(
            yt, yp, n_boot=self.bootstrap_samples, seed=self.seed
        )
        self._last_bootstrap_ci = {
            k: {"ci_lo": float(lo), "ci_hi": float(hi)} for k, (lo, hi) in ci.items()
        }
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        # Named in the report, not only in the log: a macro average
        # over a filtered set of classes has to say which set.
        report["excluded_labels"] = dict(getattr(self, "_dropped_labels", {}) or {})
        report["min_cells_per_label"] = self.min_cells_per_label
        artefacts = write_predictions(
            output_dir=self.output_dir,
            test_tokens=predictions["test_tokens"],
            test_cell_types=list(predictions["test_labels"]),
            test_tissues=predictions["test_tissue"],
            predictions=list(predictions["predictions"]),
            arch=self.handle.arch,
            preview_count=self.predictions_preview_count,
        )
        report.update(
            {
                "test_size": predictions["test_size"],
                "train_size": predictions["train_size"],
                "num_classes": int(len(set(dataset["cell_type"]))),
                "num_classes_in_test": int(len(set(predictions["test_labels"].tolist()))),
                "seed": self.seed,
                "bootstrap_ci_95": getattr(self, "_last_bootstrap_ci", {}),
                "artefacts": artefacts,
                "notes": (
                    "Cell-type classification via encoder embed + LogReg probe. "
                    "Tokens are passed straight to the adapter (no string "
                    "round-trip)."
                ),
            }
        )
        return report
