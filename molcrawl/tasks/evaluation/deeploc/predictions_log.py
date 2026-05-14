"""Per-protein predictions dump for the DeepLoc evaluator.

Two artefacts:

- ``predictions.jsonl`` — one record per test row with sequence
  truncation, accession (if present), gold class, predicted class,
  correctness flag.
- ``predictions.txt`` — narrative preview. Samples a handful of
  CORRECT vs WRONG predictions per kingdom (Eukaryota / Bacteria /
  Archaea / Virus) so the reader can spot kingdom-specific failure
  modes (e.g. eukaryote-only models failing on bacterial sequences).
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
    classes: Sequence[str],
    arch: Optional[str] = None,
    preview_count: int = 20,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    _write_jsonl(jsonl_path, test_df, y_pred, classes)
    _write_narrative(
        txt_path,
        test_df=test_df,
        y_pred=y_pred,
        classes=classes,
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
    classes: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        df = test_df.reset_index(drop=True)
        y_true = df["label"].astype(int).to_numpy() if "label" in df.columns else None
        for i, row in df.iterrows():
            pred_idx = int(y_pred[i]) if i < len(y_pred) else -1
            true_idx = int(y_true[i]) if y_true is not None else -1
            rec: Dict[str, Any] = {
                "index": int(i),
                "accession": str(row.get("accession", "")) or None,
                "kingdom": str(row.get("kingdom", "")) or None,
                "cluster_id": int(row["cluster_id"]) if "cluster_id" in df.columns else None,
                "sequence_length": int(len(str(row.get("sequence", "")))),
                "predicted": classes[pred_idx] if 0 <= pred_idx < len(classes) else None,
                "gold": classes[true_idx] if 0 <= true_idx < len(classes) else None,
                "correct": (true_idx == pred_idx) if (y_true is not None) else None,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    test_df: pd.DataFrame,
    y_pred: Sequence[int],
    classes: Sequence[str],
    arch: Optional[str],
    preview_count: int,
) -> None:
    df = test_df.reset_index(drop=True)
    n = len(df)
    y_true = df["label"].astype(int).to_numpy() if "label" in df.columns else None
    y_pred_arr = np.asarray(y_pred[:n], dtype=int)

    with path.open("w", encoding="utf-8") as fh:
        fh.write("DeepLoc per-protein narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch       : {arch or '?'}\n")
        fh.write(f"n_test     : {n}\n")
        fh.write(f"n_classes  : {len(classes)}\n")

        if y_true is not None and n:
            correct = int((y_true == y_pred_arr).sum())
            fh.write(f"correct    : {correct} / {n} (={correct/n:.3f})\n")

            # Per-kingdom and per-class breakdown
            if "kingdom" in df.columns:
                fh.write("\n-- per-kingdom accuracy --\n")
                for k, sub in df.groupby("kingdom"):
                    yt = sub["label"].astype(int).to_numpy()
                    yp = y_pred_arr[sub.index.to_numpy()]
                    if len(yt) == 0:
                        continue
                    fh.write(
                        f"  {str(k):<20s}  n={len(yt):>4d}  "
                        f"acc={(yt == yp).mean():.3f}\n"
                    )
            fh.write("\n-- per-class accuracy --\n")
            for cls_idx, cls in enumerate(classes):
                mask = (y_true == cls_idx)
                if not mask.any():
                    continue
                fh.write(
                    f"  {cls:<26s}  n={int(mask.sum()):>4d}  "
                    f"recall={(y_pred_arr[mask] == cls_idx).mean():.3f}\n"
                )

        fh.write("\n")
        fh.write(
            "Blocks below sample CORRECT and WRONG predictions per kingdom so "
            "the reader can see whether failures concentrate in one taxonomic "
            "group (a common artefact when the encoder was trained primarily on "
            "eukaryotic sequences).\n"
        )
        fh.write(_SECTION + "\n\n")

        if y_true is None:
            fh.write("(no gold labels — preview suppressed)\n")
            return

        groups: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: {"correct": [], "wrong": []}
        )
        for i in range(n):
            kingdom = str(df.iloc[i].get("kingdom", "?")) or "?"
            bucket = "correct" if int(y_true[i]) == int(y_pred_arr[i]) else "wrong"
            groups[kingdom][bucket].append(i)

        per_bucket = max(1, preview_count // 4)
        for kingdom, buckets in groups.items():
            fh.write(_SECTION + "\n")
            fh.write(f"Kingdom: {kingdom}\n")
            fh.write(_SECTION + "\n")
            for tag, indices in [("CORRECT", buckets["correct"]), ("WRONG", buckets["wrong"])]:
                fh.write(_RULE + "\n")
                fh.write(f"{tag} samples\n")
                fh.write(_RULE + "\n")
                if not indices:
                    fh.write("(none)\n")
                    continue
                for i in indices[:per_bucket]:
                    row = df.iloc[i]
                    seq = str(row.get("sequence", ""))
                    fh.write(
                        f"  idx={i}  acc={row.get('accession', '?')}  "
                        f"len={len(seq):>4d}  gold={classes[int(y_true[i])]:<26s}"
                        f"pred={classes[int(y_pred_arr[i])]}\n"
                    )
                    fh.write(f"    {seq[:80]}{'...' if len(seq) > 80 else ''}\n")
            fh.write("\n")
