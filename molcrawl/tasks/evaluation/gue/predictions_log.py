"""Per-row predictions dump for GUE evaluator.

Two artefacts:

- ``predictions.jsonl`` — one record per test row with sequence
  truncation, gold label, predicted label, correctness flag.
- ``predictions.txt`` — narrative preview. Samples CORRECT vs WRONG
  predictions per class so the reader can spot class-specific failure
  modes (e.g. promoter sub-classes confused with each other).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def write_predictions(
    output_dir: Path,
    test_df: pd.DataFrame,
    y_pred: Sequence[int],
    task_name: str,
    num_classes: int,
    sequence_column: str = "sequence",
    label_column: str = "label",
    arch: Optional[str] = None,
    preview_count: int = 16,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    _write_jsonl(jsonl_path, test_df, y_pred, sequence_column, label_column)
    _write_narrative(
        txt_path,
        test_df=test_df,
        y_pred=y_pred,
        task_name=task_name,
        num_classes=num_classes,
        sequence_column=sequence_column,
        label_column=label_column,
        arch=arch,
        preview_count=preview_count,
    )

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


def _write_jsonl(
    path: Path,
    test_df: pd.DataFrame,
    y_pred: Sequence[int],
    sequence_column: str,
    label_column: str,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        df = test_df.reset_index(drop=True)
        y_true = (
            df[label_column].astype(int).to_numpy() if label_column in df.columns else None
        )
        for i, row in df.iterrows():
            pred_idx = int(y_pred[i]) if i < len(y_pred) else -1
            true_idx = int(y_true[i]) if y_true is not None else -1
            seq = str(row.get(sequence_column, ""))
            rec: Dict[str, Any] = {
                "index": int(i),
                "sequence_length": int(len(seq)),
                "label": true_idx if y_true is not None else None,
                "prediction": pred_idx,
                "correct": (true_idx == pred_idx) if (y_true is not None) else None,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    test_df: pd.DataFrame,
    y_pred: Sequence[int],
    task_name: str,
    num_classes: int,
    sequence_column: str,
    label_column: str,
    arch: Optional[str],
    preview_count: int,
) -> None:
    df = test_df.reset_index(drop=True)
    n = len(df)
    y_true = (
        df[label_column].astype(int).to_numpy() if label_column in df.columns else None
    )
    y_pred_arr = np.asarray(y_pred[:n], dtype=int)

    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"GUE {task_name} per-row narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch       : {arch or '?'}\n")
        fh.write(f"task       : {task_name}\n")
        fh.write(f"num_classes: {num_classes}\n")
        fh.write(f"n_test     : {n}\n")

        if y_true is not None and n:
            correct = int((y_true == y_pred_arr).sum())
            fh.write(f"correct    : {correct} / {n} (={correct/n:.3f})\n")

            fh.write("\n-- per-class recall --\n")
            for cls in sorted(np.unique(y_true)):
                mask = y_true == cls
                if not mask.any():
                    continue
                fh.write(
                    f"  class={int(cls):>2d}  n={int(mask.sum()):>4d}  "
                    f"recall={(y_pred_arr[mask] == cls).mean():.3f}\n"
                )

            fh.write("\n-- confusion (rows=gold, cols=pred) --\n")
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(y_true, y_pred_arr):
                if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                    cm[int(t), int(p)] += 1
            for r in range(num_classes):
                fh.write("  " + " ".join(f"{int(cm[r, c]):>5d}" for c in range(num_classes)) + "\n")

        fh.write("\n")
        fh.write(
            "Blocks below sample CORRECT and WRONG predictions per class so "
            "failure modes (e.g. confusion between adjacent histone marks) become "
            "visible.\n"
        )
        fh.write(_SECTION + "\n\n")

        if y_true is None:
            fh.write("(no gold labels — preview suppressed)\n")
            return

        groups: Dict[int, Dict[str, List[int]]] = defaultdict(
            lambda: {"correct": [], "wrong": []}
        )
        for i in range(n):
            cls = int(y_true[i])
            bucket = "correct" if cls == int(y_pred_arr[i]) else "wrong"
            groups[cls][bucket].append(i)

        per_bucket = max(1, preview_count // (2 * max(1, num_classes)))
        for cls in sorted(groups.keys()):
            buckets = groups[cls]
            fh.write(_SECTION + "\n")
            fh.write(f"Class {cls}\n")
            fh.write(_SECTION + "\n")
            for tag, indices in [("CORRECT", buckets["correct"]), ("WRONG", buckets["wrong"])]:
                fh.write(_RULE + "\n")
                fh.write(f"{tag} samples\n")
                fh.write(_RULE + "\n")
                if not indices:
                    fh.write("(none)\n")
                    continue
                for i in indices[:per_bucket]:
                    seq = str(df.iloc[i][sequence_column])
                    fh.write(
                        f"  idx={i}  len={len(seq):>4d}  "
                        f"gold={int(y_true[i])}  pred={int(y_pred_arr[i])}\n"
                    )
                    fh.write(f"    {seq[:80]}{'...' if len(seq) > 80 else ''}\n")
            fh.write("\n")
