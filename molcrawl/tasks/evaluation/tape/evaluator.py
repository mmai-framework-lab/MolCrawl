"""Evaluator that dispatches to one TAPE sub-task at a time.

足固め upgrade adds:

- task-aware stratified subsample (class-balanced for classification,
  quantile-binned for regression) replacing ``df.head(max_examples)``.
- bootstrap 95 % CI on the active metric pack (RMSE/Spearman/Pearson
  for regression; accuracy/f1_macro/mcc for classification).
- per-row predictions log (jsonl + narrative TXT showing largest
  over/under-prediction errors for regression; per-class
  CORRECT / WRONG samples for classification).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import TAPETaskSpec, get_spec, stratified_subsample, to_frame
from .metrics import (
    bootstrap_ci,
    classification_metrics,
    contact_prediction_metrics,
    regression_metrics,
)
from .predictions_log import write_predictions
from .splits import load_splits

logger = logging.getLogger(__name__)


class TAPEEvaluator(BaseEvaluator):
    """Evaluate a protein encoder / decoder on one TAPE task."""

    task_name = "tape"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        task_dir: Path,
        task_spec: TAPETaskSpec,
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
            self.config.get("predictions_preview_count", 20)
        )

    def category(self) -> str:
        if self.task_spec.task_type == "sequence_labeling":
            return "sequence_annotation"
        if self.task_spec.task_type == "regression":
            return "property_prediction"
        return "property_prediction"

    def load_dataset(self) -> Dict[str, pd.DataFrame]:
        splits = load_splits(self.task_dir, self.task_spec.name)
        frames = {name: to_frame(records, self.task_spec) for name, records in splits.items()}
        if self.max_examples is not None:
            for split in list(frames.keys()):
                frames[split] = stratified_subsample(
                    frames[split],
                    n_examples=int(self.max_examples),
                    spec=self.task_spec,
                    seed=self.seed,
                )
        return frames

    def run_predictions(self, dataset):
        adapter = self.adapter
        spec = self.task_spec

        if spec.task_type == "sequence_labeling" and spec.name == "contact_prediction":
            train_df = dataset.get("train")
            test_df = dataset.get("test")
            if test_df is None:
                test_df = dataset.get("valid")
            if train_df is None or test_df is None:
                raise RuntimeError(
                    f"TAPE {spec.name}: need train + (valid or test) splits "
                    f"under {self.task_dir}"
                )
            return self._run_contact_prediction(adapter, train_df, test_df)

        train_df = dataset.get("train")
        test_df = dataset.get("test")
        if test_df is None:
            test_df = dataset.get("valid")
        if train_df is None or test_df is None:
            raise RuntimeError(
                f"TAPE {spec.name}: need train + (valid or test) splits "
                f"under {self.task_dir}"
            )

        if spec.task_type == "sequence_labeling":
            return self._run_sequence_labeling(adapter, spec, train_df, test_df)

        if not adapter.supports("embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot produce embeddings; "
                "TAPE evaluation requires encoder embeddings."
            )

        logger.info(
            "TAPE %s: embedding train=%d test=%d (task_type=%s)",
            spec.name,
            len(train_df),
            len(test_df),
            spec.task_type,
        )
        train_emb = np.asarray(
            adapter.embed(train_df[spec.sequence_column].astype(str).tolist()).embeddings
        )
        test_emb = np.asarray(
            adapter.embed(test_df[spec.sequence_column].astype(str).tolist()).embeddings
        )

        y_train = train_df[spec.label_column].to_numpy()
        if spec.task_type == "regression":
            from sklearn.linear_model import Ridge

            reg = Ridge(alpha=1.0)
            reg.fit(train_emb, y_train.astype(float))
            preds = reg.predict(test_emb)
        else:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=1000)
            clf.fit(train_emb, y_train.astype(int))
            preds = clf.predict(test_emb)

        return {
            "mode": "probe",
            "predictions": preds,
            "test_df": test_df,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
        }

    def _run_sequence_labeling(
        self, adapter, spec: TAPETaskSpec, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Per-residue probe: encoder embed_per_residue -> LogReg.

        Trains a single LogisticRegression head on residue-level features
        flattened across all training proteins, then applies it per-residue
        to each test protein. Special tokens (cls / eos / pad) and
        disorder-masked residues are excluded from the training and metric
        pools alike.
        """
        if not adapter.supports("per_residue_embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot produce per-residue "
                "embeddings; TAPE secondary_structure_* requires "
                "embed_per_residue support."
            )
        from sklearn.linear_model import LogisticRegression

        logger.info(
            "TAPE %s sequence_labeling: train=%d test=%d (num_classes=%d)",
            spec.name,
            len(train_df),
            len(test_df),
            spec.task_spec_num_classes if hasattr(spec, "task_spec_num_classes") else spec.num_classes,
        )

        train_seqs = train_df[spec.sequence_column].astype(str).tolist()
        test_seqs = test_df[spec.sequence_column].astype(str).tolist()
        train_labels = list(train_df[spec.label_column])
        test_labels = list(test_df[spec.label_column])
        train_masks = (
            list(train_df["valid_mask"])
            if "valid_mask" in train_df.columns
            else [[1] * len(s) for s in train_seqs]
        )
        test_masks = (
            list(test_df["valid_mask"])
            if "valid_mask" in test_df.columns
            else [[1] * len(s) for s in test_seqs]
        )

        train_per_res = adapter.embed_per_residue(train_seqs)
        test_per_res = adapter.embed_per_residue(test_seqs)

        # Stack train residues. The encoder usually adds [CLS] and [EOS] /
        # [SEP] padding, so the per-residue tensor length is len(primary)+2
        # (or +1 for some tokenizers). We align to the right by *trimming
        # to the length of the labels* — the first ``offset`` rows are the
        # leading cls/bos token(s) and we drop them.
        X_train_parts: List[Any] = []
        y_train_parts: List[int] = []
        for _i, (seq, hidden, labels, mask) in enumerate(
            zip(train_seqs, train_per_res, train_labels, train_masks)
        ):
            X_p, y_p, _ = _align_and_mask(hidden, labels, mask, seq_len=len(seq))
            if X_p.size == 0:
                continue
            X_train_parts.append(X_p)
            y_train_parts.extend(int(v) for v in y_p)
        if not X_train_parts:
            raise RuntimeError(
                "No usable training residues remained after aligning + masking."
            )
        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.asarray(y_train_parts, dtype=int)

        clf = LogisticRegression(max_iter=200, n_jobs=1)
        clf.fit(X_train, y_train)

        per_protein_pred: List[List[int]] = []
        per_protein_label: List[List[int]] = []
        per_protein_mask: List[List[int]] = []
        for seq, hidden, labels, mask in zip(
            test_seqs, test_per_res, test_labels, test_masks
        ):
            X_p, y_p, m_p = _align_and_mask(
                hidden, labels, mask, seq_len=len(seq), keep_invalid=True
            )
            if X_p.size == 0:
                # No usable rows; emit empty placeholder so length parity holds.
                per_protein_pred.append([])
                per_protein_label.append([])
                per_protein_mask.append([])
                continue
            preds = clf.predict(X_p)
            per_protein_pred.append([int(v) for v in preds])
            per_protein_label.append([int(v) for v in y_p])
            per_protein_mask.append([int(v) for v in m_p])

        return {
            "mode": "sequence_labeling",
            "per_protein_pred": per_protein_pred,
            "per_protein_label": per_protein_label,
            "per_protein_mask": per_protein_mask,
            "test_df": test_df,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
        }

    def _run_contact_prediction(
        self, adapter, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Per-residue-pair logistic-regression probe for contact prediction.

        For each protein the dataset stores a list of (i, j) pairs that
        ARE in contact (the negatives are implicit in the complement).
        We:

        1. Embed every train + test protein per-residue.
        2. Sample ``contact_pairs_per_protein`` long-range positives and
           the same number of long-range negatives from each train protein.
           Pair feature = element-wise product of the two residue
           embeddings (symmetric, dim = D).
        3. Train one shared :class:`LogisticRegression` head on the
           concatenated training pairs.
        4. For each test protein, score *every* long-range pair
           (j - i >= ``contact_min_separation``) and return the score
           array + 0/1 labels for the metric.
        """
        if not adapter.supports("per_residue_embedding"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot produce per-residue "
                "embeddings; TAPE contact_prediction requires "
                "embed_per_residue support."
            )
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(self.seed)
        min_sep = int(self.config.get("contact_min_separation", 24))
        pairs_per_train = int(self.config.get("contact_pairs_per_protein", 50))

        train_seqs = train_df["primary"].astype(str).tolist()
        test_seqs = test_df["primary"].astype(str).tolist()
        logger.info(
            "TAPE contact_prediction: embedding train=%d test=%d (min_sep=%d, "
            "pairs_per_train=%d)",
            len(train_seqs), len(test_seqs), min_sep, pairs_per_train,
        )

        train_emb = list(adapter.embed_per_residue(train_seqs))
        test_emb = list(adapter.embed_per_residue(test_seqs))

        train_X: List[np.ndarray] = []
        train_y: List[int] = []
        for emb_p, contacts_p, seq_p in zip(
            train_emb,
            train_df["tertiary"].tolist(),
            train_seqs,
        ):
            emb_arr = np.asarray(emb_p, dtype=np.float32)
            L = min(emb_arr.shape[0], len(seq_p))
            if L < min_sep + 2:
                continue
            contact_set = {(int(p[0]), int(p[1])) for p in contacts_p
                           if int(p[0]) < L and int(p[1]) < L}
            # sample positives (long-range)
            pos_pairs = [(i, j) for (i, j) in contact_set if 0 <= i < j < L and j - i >= min_sep]
            if not pos_pairs:
                continue
            n_pos = min(pairs_per_train, len(pos_pairs))
            pos_idx = rng.choice(len(pos_pairs), size=n_pos, replace=False)
            for k in pos_idx:
                i, j = pos_pairs[int(k)]
                train_X.append(emb_arr[i] * emb_arr[j])
                train_y.append(1)
            # sample negatives uniformly from long-range non-contacts
            n_neg = n_pos
            tries = 0
            taken = 0
            while taken < n_neg and tries < 20 * n_neg:
                i = int(rng.integers(0, L - min_sep))
                j = int(rng.integers(i + min_sep, L))
                tries += 1
                if (i, j) in contact_set:
                    continue
                train_X.append(emb_arr[i] * emb_arr[j])
                train_y.append(0)
                taken += 1

        if not train_X:
            raise RuntimeError(
                "TAPE contact_prediction: no usable training pairs collected — "
                "train split too small or proteins too short for min_sep="
                f"{min_sep}."
            )
        Xtr = np.stack(train_X).astype(np.float32)
        ytr = np.asarray(train_y, dtype=int)
        logger.info(
            "TAPE contact_prediction: training pairs=%d (positives=%d)",
            Xtr.shape[0], int(ytr.sum()),
        )
        clf = LogisticRegression(max_iter=500, n_jobs=1)
        clf.fit(Xtr, ytr)

        per_protein: List[Dict[str, Any]] = []
        for emb_p, contacts_p, seq_p in zip(
            test_emb,
            test_df["tertiary"].tolist(),
            test_seqs,
        ):
            emb_arr = np.asarray(emb_p, dtype=np.float32)
            L = min(emb_arr.shape[0], len(seq_p))
            if L < min_sep + 2:
                continue
            contact_set = {(int(p[0]), int(p[1])) for p in contacts_p
                           if int(p[0]) < L and int(p[1]) < L}
            # build all long-range pair indices
            ii, jj = np.triu_indices(L, k=min_sep)
            if ii.size == 0:
                continue
            X_all = emb_arr[ii] * emb_arr[jj]
            scores = clf.predict_proba(X_all)[:, 1]
            labels = np.array(
                [1 if (int(i), int(j)) in contact_set else 0 for i, j in zip(ii, jj)],
                dtype=int,
            )
            per_protein.append({
                "seq_len": int(L),
                "pair_idx": np.stack([ii, jj], axis=1),
                "scores": scores.astype(np.float32),
                "labels": labels,
            })

        return {
            "mode": "contact_prediction",
            "per_protein": per_protein,
            "test_df": test_df,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "min_separation": min_sep,
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        spec = self.task_spec
        if predictions.get("mode") == "contact_prediction":
            return self._compute_contact_prediction_metrics(predictions)
        if predictions.get("mode") == "placeholder":
            self._last_bootstrap_ci: Dict[str, Any] = {}
            return contact_prediction_metrics([])
        if predictions.get("mode") == "sequence_labeling":
            return self._compute_sequence_labeling_metrics(predictions)
        preds = predictions["predictions"]
        test_df = predictions["test_df"]
        y_true = test_df[spec.label_column].to_numpy()
        if spec.task_type == "regression":
            yt = y_true.astype(float)
            yp = np.asarray(preds, dtype=float)
            metrics = regression_metrics(yt, yp)
        else:
            yt = y_true.astype(int)
            yp = np.asarray(preds, dtype=int)
            metrics = classification_metrics(yt, yp)
        ci = bootstrap_ci(
            yt,
            yp,
            task_type=spec.task_type,
            n_boot=self.bootstrap_samples,
            seed=self.seed,
        )
        self._last_bootstrap_ci = {
            k: {"ci_lo": float(lo), "ci_hi": float(hi)} for k, (lo, hi) in ci.items()
        }
        return metrics

    def _compute_contact_prediction_metrics(self, predictions) -> Dict[str, float]:
        from .metrics import bootstrap_contact_ci

        per_protein = predictions["per_protein"]
        min_sep = int(predictions.get("min_separation", 24))
        ks = (1, 2, 5)
        metrics = contact_prediction_metrics(
            per_protein, min_separation=min_sep, ks=ks
        )
        ci = bootstrap_contact_ci(
            per_protein,
            min_separation=min_sep,
            ks=ks,
            n_boot=self.bootstrap_samples,
            seed=self.seed,
        )
        self._last_bootstrap_ci = {
            k: {"ci_lo": float(lo), "ci_hi": float(hi)} for k, (lo, hi) in ci.items()
        }
        return metrics

    def _compute_sequence_labeling_metrics(self, predictions):
        from .metrics import (
            bootstrap_sequence_labeling_ci,
            sequence_labeling_metrics,
        )

        per_pred = predictions["per_protein_pred"]
        per_label = predictions["per_protein_label"]
        per_mask = predictions["per_protein_mask"]
        metrics = sequence_labeling_metrics(per_pred, per_label, per_mask)
        ci = bootstrap_sequence_labeling_ci(
            per_pred,
            per_label,
            per_mask,
            n_boot=self.bootstrap_samples,
            seed=self.seed,
        )
        self._last_bootstrap_ci = {
            k: {"ci_lo": float(lo), "ci_hi": float(hi)} for k, (lo, hi) in ci.items()
        }
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "task_spec": {
                    "name": self.task_spec.name,
                    "task_type": self.task_spec.task_type,
                    "label_column": self.task_spec.label_column,
                },
                "split_sizes": {k: int(len(v)) for k, v in dataset.items()},
                "seed": self.seed,
                "bootstrap_ci_95": getattr(self, "_last_bootstrap_ci", {}),
            }
        )
        if predictions.get("mode") == "probe":
            artefacts = write_predictions(
                output_dir=self.output_dir,
                test_df=predictions["test_df"],
                y_pred=predictions["predictions"],
                task_name=self.task_spec.name,
                task_type=self.task_spec.task_type,
                sequence_column=self.task_spec.sequence_column,
                label_column=self.task_spec.label_column,
                arch=self.handle.arch,
                preview_count=self.predictions_preview_count,
            )
            report["artefacts"] = artefacts
            report["train_size_used"] = predictions["train_size"]
            report["test_size_used"] = predictions["test_size"]
        elif predictions.get("mode") == "sequence_labeling":
            from .predictions_log import write_sequence_labeling_predictions

            artefacts = write_sequence_labeling_predictions(
                output_dir=self.output_dir,
                test_df=predictions["test_df"],
                per_protein_pred=predictions["per_protein_pred"],
                per_protein_label=predictions["per_protein_label"],
                per_protein_mask=predictions["per_protein_mask"],
                task_name=self.task_spec.name,
                num_classes=self.task_spec.num_classes,
                sequence_column=self.task_spec.sequence_column,
                arch=self.handle.arch,
                preview_count=self.predictions_preview_count,
            )
            report["artefacts"] = artefacts
            report["train_size_used"] = predictions["train_size"]
            report["test_size_used"] = predictions["test_size"]
        elif predictions.get("mode") == "contact_prediction":
            artefacts = self._write_contact_predictions(predictions)
            report["artefacts"] = artefacts
            report["train_size_used"] = predictions["train_size"]
            report["test_size_used"] = predictions["test_size"]
            report["min_separation"] = predictions["min_separation"]
        return report

    def _write_contact_predictions(self, predictions) -> Dict[str, str]:
        """Emit predictions.jsonl + a short narrative for contact_prediction.

        The JSONL has one record per test protein with the ranked top
        ``L // 5`` long-range pairs (the same set that the precision@L/5
        metric uses) — small enough to stay readable yet enough to spot
        whether the model preferentially picked clustered residues vs.
        well-separated ones.
        """
        import json as _json

        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        per_protein = predictions["per_protein"]
        min_sep = int(predictions["min_separation"])
        jsonl_path = out_dir / "predictions.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for i, prot in enumerate(per_protein):
                L = int(prot["seq_len"])
                top_n = max(1, L // 5)
                scores = np.asarray(prot["scores"], dtype=float)
                pair_idx = np.asarray(prot["pair_idx"], dtype=int)
                labels = np.asarray(prot["labels"], dtype=int)
                order = np.argsort(-scores)[:top_n]
                top_pairs = [
                    {
                        "i": int(pair_idx[k, 0]),
                        "j": int(pair_idx[k, 1]),
                        "score": float(scores[k]),
                        "label": int(labels[k]),
                    }
                    for k in order
                ]
                fh.write(_json.dumps({
                    "index": i,
                    "seq_len": L,
                    "min_separation": min_sep,
                    "n_long_range_pairs_scored": int(scores.size),
                    "n_true_long_range_contacts": int(labels.sum()),
                    "top_L_over_5": top_pairs,
                }) + "\n")

        txt_path = out_dir / "predictions.txt"
        with txt_path.open("w", encoding="utf-8") as fh:
            fh.write("TAPE contact_prediction summary\n")
            fh.write("=" * 72 + "\n")
            fh.write(f"arch              : {self.handle.arch}\n")
            fh.write(f"modality          : {self.handle.modality}\n")
            fh.write(f"min_separation    : {min_sep}\n")
            fh.write(f"n_proteins_scored : {len(per_protein)}\n")
            if per_protein:
                lens = [int(p["seq_len"]) for p in per_protein]
                fh.write(
                    f"seq_len           : min={min(lens)} median={int(np.median(lens))} max={max(lens)}\n"
                )
                preview_n = min(self.predictions_preview_count, len(per_protein))
                fh.write(
                    f"\nFirst {preview_n} proteins (top L/5 long-range pairs each):\n"
                )
                fh.write("-" * 72 + "\n")
                for p in per_protein[:preview_n]:
                    L = int(p["seq_len"])
                    top_n = max(1, L // 5)
                    scores = np.asarray(p["scores"], dtype=float)
                    pair_idx = np.asarray(p["pair_idx"], dtype=int)
                    labels = np.asarray(p["labels"], dtype=int)
                    order = np.argsort(-scores)[:top_n]
                    correct = int(labels[order].sum())
                    fh.write(
                        f"  L={L:>4d}  top_{top_n:>3d}  precision_L/5={correct/top_n:.3f}  "
                        f"true_LR_contacts={int(labels.sum())}\n"
                    )
        return {
            "predictions_jsonl": str(jsonl_path),
            "predictions_txt": str(txt_path),
        }


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _align_and_mask(
    hidden: np.ndarray,
    labels: List[int],
    mask: List[int],
    seq_len: int,
    keep_invalid: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align per-residue hidden states to labels + drop masked positions.

    Encoders typically prepend ``<cls>`` and append ``<eos>`` / ``<sep>``
    around the amino-acid stream, so ``hidden.shape[0] == seq_len + offset``.
    We slice off the leading ``offset`` rows so the residue at index *i*
    in ``hidden_aligned`` matches label index *i*.

    When ``keep_invalid`` is False (training pool) we return only the
    valid residues. When True (test pool) we keep length parity with
    ``labels`` and surface the mask alongside.
    """
    hidden = np.asarray(hidden)
    if hidden.ndim != 2:
        return np.zeros((0, 0)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    L = hidden.shape[0]
    if L == 0:
        return np.zeros((0, hidden.shape[1] if hidden.ndim == 2 else 0)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    offset = L - seq_len
    if offset < 0:
        # encoder sequence got truncated below the label length; trim
        # labels instead so we line up correctly on the prefix.
        seq_len = L
        labels = labels[:L]
        mask = mask[:L]
        offset = 0
    aligned = hidden[offset : offset + seq_len, :]
    labels_arr = np.asarray(labels[:seq_len], dtype=int)
    mask_arr = np.asarray(mask[:seq_len], dtype=int)

    if keep_invalid:
        return aligned, labels_arr, mask_arr
    keep = mask_arr.astype(bool)
    if not keep.any():
        return np.zeros((0, hidden.shape[1])), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return aligned[keep], labels_arr[keep], mask_arr[keep]


def evaluate_task(handle: ModelHandle, task_name: str, **kwargs: Any):
    spec = get_spec(task_name)
    evaluator = TAPEEvaluator(handle=handle, task_spec=spec, **kwargs)
    return evaluator.run()
