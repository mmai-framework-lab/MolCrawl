"""Per-test-row prediction dumps for the MoleculeNet evaluator.

Emits two artefacts alongside ``metrics.json`` / ``REPORT.md``:

- ``predictions.jsonl`` — one record per test row with SMILES, label(s),
  model score(s), and (for regression) the absolute error. Machine
  readable; feeds downstream plotting / error analysis.
- ``predictions.txt`` — narrative preview. For classification, samples
  both top-ranked and bottom-ranked molecules under the probe's score
  along with the ground-truth class. For regression, samples the
  largest over- and under-predictions to visualise where the probe
  falls short.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def write_predictions(
    output_dir: Path,
    test_df: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    label_columns: Sequence[str],
    smiles_column: str,
    task_type: str,
    mode: str,
    split_sizes: Optional[Dict[str, int]] = None,
    arch: Optional[str] = None,
    preview_count: int = 20,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    _write_jsonl(
        jsonl_path,
        test_df=test_df,
        preds=preds,
        label_columns=label_columns,
        smiles_column=smiles_column,
        task_type=task_type,
        mode=mode,
    )

    txt_path = output_dir / "predictions.txt"
    _write_narrative(
        txt_path,
        test_df=test_df,
        preds=preds,
        label_columns=label_columns,
        smiles_column=smiles_column,
        task_type=task_type,
        mode=mode,
        split_sizes=split_sizes,
        arch=arch,
        preview_count=preview_count,
    )

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


# ---------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------


def _write_jsonl(
    path: Path,
    test_df: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    label_columns: Sequence[str],
    smiles_column: str,
    task_type: str,
    mode: str,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, row in test_df.reset_index(drop=True).iterrows():
            record: Dict[str, Any] = {
                "index": int(i),
                "smiles": str(row.get(smiles_column, "")),
                "mode": mode,
                "task_type": task_type,
            }
            for col in label_columns:
                val = row.get(col)
                record[f"label.{col}"] = (
                    None
                    if val is None or (isinstance(val, float) and math.isnan(val))
                    else float(val) if task_type == "regression" else int(val)
                )
                if col in preds:
                    score = preds[col][i]
                    record[f"pred.{col}"] = (
                        None if math.isnan(float(score)) else float(score)
                    )
                    if task_type == "regression" and record[f"label.{col}"] is not None:
                        record[f"abs_error.{col}"] = abs(
                            record[f"label.{col}"] - float(score)
                        )
            if mode == "zero_shot_likelihood" and "log_likelihood" in preds:
                record["log_likelihood"] = float(preds["log_likelihood"][i])
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    test_df: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    label_columns: Sequence[str],
    smiles_column: str,
    task_type: str,
    mode: str,
    split_sizes: Optional[Dict[str, int]],
    arch: Optional[str],
    preview_count: int,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        _write_preamble(
            fh,
            test_df=test_df,
            label_columns=label_columns,
            task_type=task_type,
            mode=mode,
            split_sizes=split_sizes,
            arch=arch,
        )
        if mode == "zero_shot_likelihood":
            _write_perplexity_preview(fh, test_df, preds, smiles_column, preview_count)
        else:
            _write_probe_preview(
                fh,
                test_df=test_df,
                preds=preds,
                label_columns=label_columns,
                smiles_column=smiles_column,
                task_type=task_type,
                preview_count=preview_count,
            )


def _write_preamble(
    fh,
    test_df: pd.DataFrame,
    label_columns: Sequence[str],
    task_type: str,
    mode: str,
    split_sizes: Optional[Dict[str, int]],
    arch: Optional[str],
) -> None:
    fh.write("MoleculeNet per-test-row prediction narrative\n")
    fh.write(_SECTION + "\n")
    fh.write(f"arch            : {arch or '?'}\n")
    fh.write(f"mode            : {mode}\n")
    fh.write(f"task_type       : {task_type}\n")
    fh.write(f"n_test          : {len(test_df)}\n")
    fh.write(f"label_columns   : {list(label_columns)}\n")
    if split_sizes:
        fh.write(
            f"split_sizes     : train={split_sizes.get('train', 0)} "
            f"val={split_sizes.get('val', 0)} "
            f"test={split_sizes.get('test', 0)}\n"
        )

    fh.write("\n-- test-set label distribution --\n")
    for col in label_columns:
        s = pd.Series(test_df[col])
        valid = s.dropna()
        if valid.empty:
            fh.write(f"  {col:30s}  n=0  (all NaN)\n")
            continue
        if task_type == "classification":
            vc = valid.astype(int).value_counts().sort_index().to_dict()
            fh.write(f"  {col:30s}  n={len(valid):>5d}  classes={vc}\n")
        else:
            arr = valid.astype(float)
            fh.write(
                f"  {col:30s}  n={len(valid):>5d}  "
                f"mean={arr.mean():+.3f} std={arr.std():.3f} "
                f"min={arr.min():+.3f} max={arr.max():+.3f}\n"
            )

    if mode == "zero_shot_likelihood":
        fh.write(
            "\nNote: adapter supports likelihood only; reporting mean perplexity of the test SMILES "
            "instead of a probe-based property prediction. This is a language-model-quality signal "
            "rather than a task-performance signal.\n"
        )
    fh.write(_SECTION + "\n\n")


def _write_probe_preview(
    fh,
    test_df: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    label_columns: Sequence[str],
    smiles_column: str,
    task_type: str,
    preview_count: int,
) -> None:
    if preview_count <= 0:
        return
    # Preview strategy: pick up to ``preview_count`` rows split across the
    # first label column. For classification, show the highest- and
    # lowest-scored molecules per class. For regression, show the
    # largest over- and under-predictions.
    col = label_columns[0]
    if col not in preds:
        fh.write(f"(no probe predictions available for {col})\n")
        return
    scores = np.asarray(preds[col], dtype=float)
    labels = pd.Series(test_df[col]).to_numpy()
    smis = test_df[smiles_column].astype(str).tolist()

    if task_type == "classification":
        indices = _pick_classification_preview(labels, scores, preview_count)
        for rank, i in enumerate(indices, start=1):
            _write_class_row(
                fh, rank, len(indices), i, smi=smis[i], label=labels[i], score=scores[i]
            )
    else:
        indices = _pick_regression_preview(labels, scores, preview_count)
        for rank, i in enumerate(indices, start=1):
            _write_reg_row(
                fh, rank, len(indices), i, smi=smis[i], label=labels[i], pred=scores[i]
            )


def _pick_classification_preview(
    labels: np.ndarray, scores: np.ndarray, preview_count: int
) -> List[int]:
    valid_mask = ~pd.isna(labels) & ~pd.isna(scores)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return []
    order = valid_idx[np.argsort(scores[valid_idx])]
    top_k = min(preview_count // 2, len(order))
    bot_k = min(preview_count - top_k, len(order) - top_k)
    top = order[-top_k:][::-1]
    bot = order[:bot_k]
    picked = list(bot) + list(top)
    return sorted(set(int(x) for x in picked))


def _pick_regression_preview(
    labels: np.ndarray, preds: np.ndarray, preview_count: int
) -> List[int]:
    label_arr = pd.to_numeric(labels, errors="coerce")
    pred_arr = pd.to_numeric(preds, errors="coerce")
    err = label_arr - pred_arr
    valid_mask = ~np.isnan(err)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return []
    order = valid_idx[np.argsort(err[valid_idx])]
    top_k = min(preview_count // 2, len(order))
    bot_k = min(preview_count - top_k, len(order) - top_k)
    top = order[-top_k:][::-1]  # largest positive error → model under-predicted
    bot = order[:bot_k]          # largest negative error → model over-predicted
    picked = list(bot) + list(top)
    return sorted(set(int(x) for x in picked))


def _write_class_row(
    fh, rank: int, total: int, i: int, smi: str, label: Any, score: float
) -> None:
    label_str = "?" if label is None or (isinstance(label, float) and math.isnan(label)) else f"{int(label)}"
    correct = (
        "?"
        if label_str == "?"
        else ("✓" if (int(label) == int(score >= 0.5)) else "✗")
    )
    fh.write(_SECTION + "\n")
    fh.write(f"Example {rank} of {total} (test index {i})\n")
    fh.write(f"SMILES      : {smi}\n")
    fh.write(
        f"Probe score : {score:+.4f}   pred={int(score >= 0.5)}   label={label_str}   {correct}\n"
    )
    fh.write(_SECTION + "\n\n")


def _write_reg_row(
    fh, rank: int, total: int, i: int, smi: str, label: Any, pred: float
) -> None:
    if label is None or (isinstance(label, float) and math.isnan(label)):
        label_str, err_str = "?", "?"
    else:
        label_f = float(label)
        err = label_f - float(pred)
        label_str = f"{label_f:+.3f}"
        err_str = f"{err:+.3f}"
    fh.write(_SECTION + "\n")
    fh.write(f"Example {rank} of {total} (test index {i})\n")
    fh.write(f"SMILES      : {smi}\n")
    fh.write(
        f"Prediction  : pred={pred:+.3f}   label={label_str}   error={err_str}\n"
    )
    fh.write(_SECTION + "\n\n")


def _write_perplexity_preview(
    fh,
    test_df: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    smiles_column: str,
    preview_count: int,
) -> None:
    if preview_count <= 0:
        return
    ll = np.asarray(preds.get("log_likelihood", []), dtype=float)
    if len(ll) == 0:
        fh.write("(no log-likelihood preview available)\n")
        return
    smis = test_df[smiles_column].astype(str).tolist()
    order = np.argsort(ll)
    top_k = min(preview_count // 2, len(order))
    bot_k = min(preview_count - top_k, len(order) - top_k)
    picked = list(order[:bot_k]) + list(order[-top_k:][::-1])
    for rank, i in enumerate(sorted(set(int(x) for x in picked)), start=1):
        fh.write(_SECTION + "\n")
        fh.write(
            f"Example {rank} of {preview_count} (test index {i})\n"
            f"SMILES      : {smis[i]}\n"
            f"LL          : {float(ll[i]):+.4f}\n"
        )
        fh.write(_SECTION + "\n\n")
