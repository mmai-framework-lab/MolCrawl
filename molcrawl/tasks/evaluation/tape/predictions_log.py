"""Per-row predictions dump for TAPE evaluator.

Three artefacts depending on task mode:

- regression / classification: ``predictions.jsonl`` (one record per
  test row) + ``predictions.txt`` narrative (over/under-predicted
  for regression; CORRECT vs WRONG per class for classification).
- sequence_labeling (secondary_structure_*): ``predictions.jsonl``
  one record per protein with primary / gold-ss / pred-ss strings +
  per-residue accuracy; ``predictions.txt`` shows worst/best-fit
  proteins so confusion patterns surface as text alignments.
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
    y_pred: Sequence[float],
    task_name: str,
    task_type: str,
    sequence_column: str = "primary",
    label_column: str = "log_fluorescence",
    arch: Optional[str] = None,
    preview_count: int = 20,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    _write_jsonl(jsonl_path, test_df, y_pred, task_type, sequence_column, label_column)
    _write_narrative(
        txt_path,
        test_df=test_df,
        y_pred=y_pred,
        task_name=task_name,
        task_type=task_type,
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
    y_pred: Sequence[float],
    task_type: str,
    sequence_column: str,
    label_column: str,
) -> None:
    df = test_df.reset_index(drop=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, row in df.iterrows():
            seq = str(row.get(sequence_column, ""))
            gold = row.get(label_column)
            pred_v = y_pred[i] if i < len(y_pred) else None
            rec: Dict[str, Any] = {
                "index": int(i),
                "sequence_length": int(len(seq)),
                "label": _to_jsonable(gold),
                "prediction": _to_jsonable(pred_v),
            }
            if task_type == "regression" and gold is not None and pred_v is not None:
                try:
                    rec["abs_error"] = float(abs(float(gold) - float(pred_v)))
                except Exception:
                    pass
            elif task_type in ("classification", "sequence_labeling") and gold is not None and pred_v is not None:
                try:
                    rec["correct"] = int(gold) == int(pred_v)
                except Exception:
                    pass
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    test_df: pd.DataFrame,
    y_pred: Sequence[float],
    task_name: str,
    task_type: str,
    sequence_column: str,
    label_column: str,
    arch: Optional[str],
    preview_count: int,
) -> None:
    df = test_df.reset_index(drop=True)
    n = len(df)
    y_true = df[label_column].to_numpy() if label_column in df.columns else None
    y_pred_arr = np.asarray(y_pred[:n])

    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"TAPE {task_name} per-row narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch       : {arch or '?'}\n")
        fh.write(f"task       : {task_name}\n")
        fh.write(f"task_type  : {task_type}\n")
        fh.write(f"n_test     : {n}\n")

        if y_true is None:
            fh.write("(no gold labels — preview suppressed)\n")
            return

        if task_type == "regression":
            yt = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy()
            yp = pd.to_numeric(pd.Series(y_pred_arr), errors="coerce").to_numpy()
            valid = ~(np.isnan(yt) | np.isnan(yp))
            if valid.any():
                fh.write(
                    f"label spread: gold min={yt[valid].min():+.3f} "
                    f"max={yt[valid].max():+.3f} mean={yt[valid].mean():+.3f}\n"
                )
                fh.write(
                    f"pred spread : pred min={yp[valid].min():+.3f} "
                    f"max={yp[valid].max():+.3f} mean={yp[valid].mean():+.3f}\n"
                )
            fh.write("\n")
            fh.write(
                "Block A samples the largest UNDER-predictions (model output much "
                "lower than gold). Block B samples the largest OVER-predictions. "
                "Both reveal saturation / poor extrapolation.\n"
            )
            fh.write(_SECTION + "\n\n")
            _regression_preview(fh, df, yt, yp, sequence_column, preview_count)
        else:
            yt_i = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy()
            yp_i = pd.to_numeric(pd.Series(y_pred_arr), errors="coerce").to_numpy()
            valid = ~(np.isnan(yt_i) | np.isnan(yp_i))
            if valid.any():
                correct = int((yt_i[valid] == yp_i[valid]).sum())
                fh.write(f"correct    : {correct} / {int(valid.sum())} "
                         f"(={correct/int(valid.sum()):.3f})\n")
            fh.write("\n")
            fh.write(
                "Blocks below sample CORRECT and WRONG predictions for the most "
                "frequent gold classes; classification confusion shows up here.\n"
            )
            fh.write(_SECTION + "\n\n")
            _classification_preview(fh, df, yt_i, yp_i, sequence_column, preview_count)


def _regression_preview(
    fh, df, yt: np.ndarray, yp: np.ndarray, sequence_column: str, preview_count: int
) -> None:
    err = yt - yp
    valid_mask = ~np.isnan(err)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        fh.write("(no valid label / prediction pairs)\n")
        return
    order = valid_idx[np.argsort(err[valid_idx])]
    half = max(1, preview_count // 2)
    bot = list(order[:half])           # err = yt - yp very negative → over-predict
    top = list(order[-half:][::-1])    # very positive → under-predict

    rank = 0
    for label, indices in [
        ("OVER-predicted (gold << prediction)", bot),
        ("UNDER-predicted (gold >> prediction)", top),
    ]:
        fh.write(_SECTION + "\n")
        fh.write(f"{label}\n")
        fh.write(_SECTION + "\n")
        for i in indices:
            rank += 1
            seq = str(df.iloc[i][sequence_column])
            fh.write(_RULE + "\n")
            fh.write(
                f"Example {rank}  idx={i}  len={len(seq):>4d}  "
                f"gold={float(yt[i]):+.3f}  pred={float(yp[i]):+.3f}  "
                f"err={float(err[i]):+.3f}\n"
            )
            fh.write(f"  {seq[:80]}{'...' if len(seq) > 80 else ''}\n")
        fh.write("\n")


def _classification_preview(
    fh, df, yt: np.ndarray, yp: np.ndarray, sequence_column: str, preview_count: int
) -> None:
    valid_mask = ~(np.isnan(yt) | np.isnan(yp))
    if not valid_mask.any():
        fh.write("(no valid label / prediction pairs)\n")
        return

    classes, counts = np.unique(yt[valid_mask].astype(int), return_counts=True)
    order_classes = classes[np.argsort(-counts)][:5]
    per_class = max(1, preview_count // (2 * max(1, len(order_classes))))

    for cls in order_classes:
        mask = valid_mask & (yt == cls)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        correct = idxs[yp[idxs] == cls]
        wrong = idxs[yp[idxs] != cls]

        fh.write(_SECTION + "\n")
        fh.write(f"Class {int(cls)}  (n_test={int(idxs.size)})\n")
        fh.write(_SECTION + "\n")
        for tag, pool in [("CORRECT", correct), ("WRONG", wrong)]:
            fh.write(_RULE + "\n")
            fh.write(f"{tag} samples\n")
            fh.write(_RULE + "\n")
            if pool.size == 0:
                fh.write("(none)\n")
                continue
            for i in pool[:per_class]:
                seq = str(df.iloc[int(i)][sequence_column])
                fh.write(
                    f"  idx={int(i)}  len={len(seq):>4d}  "
                    f"gold={int(yt[int(i)])}  pred={int(yp[int(i)])}\n"
                )
                fh.write(f"  {seq[:80]}{'...' if len(seq) > 80 else ''}\n")
        fh.write("\n")


# ---------------------------------------------------------------------
# sequence_labeling artefacts (secondary_structure_*)
# ---------------------------------------------------------------------


_DSSP3_ALPHABET = "CEH"
_DSSP8_ALPHABET = "BCEGHIST"


def _alphabet_for(num_classes: int) -> str:
    if num_classes == 3:
        return _DSSP3_ALPHABET
    if num_classes == 8:
        return _DSSP8_ALPHABET
    return "".join(str(i) for i in range(num_classes))


def write_sequence_labeling_predictions(
    output_dir: Path,
    test_df: pd.DataFrame,
    per_protein_pred: Sequence[Sequence[int]],
    per_protein_label: Sequence[Sequence[int]],
    per_protein_mask: Sequence[Sequence[int]],
    task_name: str,
    num_classes: int,
    sequence_column: str = "primary",
    arch: Optional[str] = None,
    preview_count: int = 12,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    df = test_df.reset_index(drop=True)
    seqs = df[sequence_column].astype(str).tolist()
    alphabet = _alphabet_for(num_classes)

    rows: List[Dict[str, Any]] = []
    for i in range(len(seqs)):
        if i >= len(per_protein_pred):
            break
        pred = list(per_protein_pred[i])
        label = list(per_protein_label[i])
        mask = list(per_protein_mask[i])
        n_valid = int(sum(1 for m in mask if m))
        if n_valid == 0:
            acc = float("nan")
        else:
            acc = float(
                sum(1 for j, m in enumerate(mask) if m and pred[j] == label[j])
                / n_valid
            )
        gold_str = "".join(
            alphabet[label[j]] if 0 <= label[j] < len(alphabet) else "-"
            for j in range(len(label))
        )
        pred_str = "".join(
            alphabet[pred[j]] if 0 <= pred[j] < len(alphabet) else "-"
            for j in range(len(pred))
        )
        rows.append(
            {
                "index": int(i),
                "primary": seqs[i],
                "gold_ss": gold_str,
                "pred_ss": pred_str,
                "valid_residues": n_valid,
                "per_residue_accuracy": acc,
            }
        )

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write_seq_labeling_narrative(
        txt_path,
        rows=rows,
        task_name=task_name,
        num_classes=num_classes,
        arch=arch,
        preview_count=preview_count,
    )

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


def _write_seq_labeling_narrative(
    path: Path,
    rows: List[Dict[str, Any]],
    task_name: str,
    num_classes: int,
    arch: Optional[str],
    preview_count: int,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"TAPE {task_name} per-protein narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch        : {arch or '?'}\n")
        fh.write(f"task        : {task_name}\n")
        fh.write(f"num_classes : {num_classes}\n")
        fh.write(f"n_test      : {len(rows)}\n")

        valid_rows = [r for r in rows if not math.isnan(r["per_residue_accuracy"])]
        if valid_rows:
            accs = [r["per_residue_accuracy"] for r in valid_rows]
            fh.write(
                f"per-protein acc: mean={float(np.mean(accs)):.3f} "
                f"min={float(min(accs)):.3f} max={float(max(accs)):.3f} "
                f"(scored={len(valid_rows)} of {len(rows)})\n"
            )

        # per-class recall (over all valid residues across proteins)
        cls_total: Dict[int, int] = {}
        cls_correct: Dict[int, int] = {}
        for r in rows:
            for g, p in zip(r["gold_ss"], r["pred_ss"]):
                if g == "-":
                    continue
                cls_total[g] = cls_total.get(g, 0) + 1
                if g == p:
                    cls_correct[g] = cls_correct.get(g, 0) + 1
        if cls_total:
            fh.write("\n-- per-class residue recall --\n")
            for cls in sorted(cls_total.keys()):
                got = cls_correct.get(cls, 0)
                total = cls_total[cls]
                fh.write(f"  {cls}: n={total:>5d}  recall={got/total:.3f}\n")

        fh.write("\n")
        fh.write(
            "Block A samples WORST-fit proteins (lowest per-residue accuracy); "
            "Block B samples BEST-fit. Each block prints the primary sequence, "
            "the gold secondary-structure string, and the predicted string, all "
            "wrapped at 80 chars so visual mismatches stand out.\n"
        )
        fh.write(_SECTION + "\n\n")

        scored = list(valid_rows)
        if not scored:
            fh.write("(no scored proteins)\n")
            return
        scored.sort(key=lambda r: r["per_residue_accuracy"])
        half = max(1, preview_count // 2)
        bot = scored[:half]
        top = list(reversed(scored[-half:]))

        for label, items in [("WORST-FIT", bot), ("BEST-FIT", top)]:
            fh.write(_SECTION + "\n")
            fh.write(f"{label} samples\n")
            fh.write(_SECTION + "\n")
            for r in items:
                fh.write(_RULE + "\n")
                fh.write(
                    f"index={r['index']}  acc={r['per_residue_accuracy']:.3f}  "
                    f"valid={r['valid_residues']}  len={len(r['primary'])}\n"
                )
                _wrap_three(fh, r["primary"], r["gold_ss"], r["pred_ss"])
            fh.write("\n")


def _wrap_three(fh, primary: str, gold: str, pred: str, width: int = 80) -> None:
    n = max(len(primary), len(gold), len(pred))
    for i in range(0, n, width):
        seg_p = primary[i : i + width]
        seg_g = gold[i : i + width]
        seg_pred = pred[i : i + width]
        fh.write(f"  primary [{i:>4d}] : {seg_p}\n")
        fh.write(f"  gold    [{i:>4d}] : {seg_g}\n")
        fh.write(f"  pred    [{i:>4d}] : {seg_pred}\n")
        # mark mismatches with a caret line
        marker = "".join(
            " " if (g == p or g == "-") else "^"
            for g, p in zip(seg_g, seg_pred + " " * (len(seg_g) - len(seg_pred)))
        )
        fh.write(f"  diff    [{i:>4d}] : {marker}\n")
