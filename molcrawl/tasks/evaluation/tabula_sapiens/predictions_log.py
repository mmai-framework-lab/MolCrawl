"""Per-cell predictions dump for the Tabula Sapiens evaluator.

Two artefacts:

- ``predictions.jsonl`` — one record per held-out cell with tissue,
  number of tokens, gold and predicted cell-type labels, correctness.
- ``predictions.txt`` — narrative preview. Per-class CORRECT vs WRONG
  samples for the most-frequent classes so confusion patterns surface.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def write_predictions(
    output_dir: Path,
    test_tokens: Sequence[Sequence[int]],
    test_cell_types: Sequence[str],
    test_tissues: Sequence[str],
    predictions: Sequence[str],
    arch: Optional[str] = None,
    preview_count: int = 16,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    _write_jsonl(jsonl_path, test_tokens, test_cell_types, test_tissues, predictions)
    _write_narrative(
        txt_path,
        test_tokens=test_tokens,
        test_cell_types=test_cell_types,
        test_tissues=test_tissues,
        predictions=predictions,
        arch=arch,
        preview_count=preview_count,
    )

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


def _write_jsonl(
    path: Path,
    test_tokens: Sequence[Sequence[int]],
    test_cell_types: Sequence[str],
    test_tissues: Sequence[str],
    predictions: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, (tokens, gold, tissue, pred) in enumerate(
            zip(test_tokens, test_cell_types, test_tissues, predictions)
        ):
            rec: Dict[str, Any] = {
                "index": int(i),
                "n_tokens": int(len(tokens)),
                "tissue": str(tissue),
                "gold": str(gold),
                "prediction": str(pred),
                "correct": bool(str(gold) == str(pred)),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    test_tokens: Sequence[Sequence[int]],
    test_cell_types: Sequence[str],
    test_tissues: Sequence[str],
    predictions: Sequence[str],
    arch: Optional[str],
    preview_count: int,
) -> None:
    n = len(test_cell_types)
    correct = sum(
        1 for g, p in zip(test_cell_types, predictions) if str(g) == str(p)
    )

    with path.open("w", encoding="utf-8") as fh:
        fh.write("Tabula Sapiens per-cell narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch       : {arch or '?'}\n")
        fh.write(f"n_test     : {n}\n")
        if n:
            fh.write(f"correct    : {correct} / {n} (={correct/n:.3f})\n")

        # Per-class accuracy
        class_correct: Dict[str, int] = {}
        class_total: Dict[str, int] = {}
        for g, p in zip(test_cell_types, predictions):
            class_total[str(g)] = class_total.get(str(g), 0) + 1
            if str(g) == str(p):
                class_correct[str(g)] = class_correct.get(str(g), 0) + 1
        if class_total:
            fh.write("\n-- per-class recall --\n")
            for cls, total in sorted(class_total.items(), key=lambda kv: -kv[1]):
                got = class_correct.get(cls, 0)
                fh.write(
                    f"  {cls[:36]:<36s}  n={total:>4d}  recall={got/total:.3f}\n"
                )

        # Tissue distribution (passthrough)
        if any(test_tissues):
            tissue_counts = Counter(test_tissues)
            fh.write("\n-- tissue distribution --\n")
            for t, c in tissue_counts.most_common():
                fh.write(f"  {t:<30s}  n={c}\n")

        fh.write("\n")
        fh.write(
            "Blocks below sample CORRECT and WRONG predictions for the most "
            "frequent gold classes so confusion between adjacent cell types "
            "is visible.\n"
        )
        fh.write(_SECTION + "\n\n")

        if n == 0:
            fh.write("(no test cells)\n")
            return

        # Bucket indices per gold class, then sample most-frequent
        bucket: Dict[str, Dict[str, List[int]]] = {}
        for i, (g, p) in enumerate(zip(test_cell_types, predictions)):
            d = bucket.setdefault(str(g), {"correct": [], "wrong": []})
            d["correct" if str(g) == str(p) else "wrong"].append(i)
        ordered_classes = sorted(class_total.items(), key=lambda kv: -kv[1])[:5]
        per_bucket = max(1, preview_count // (2 * max(1, len(ordered_classes))))
        for cls, _count in ordered_classes:
            buckets = bucket.get(cls, {"correct": [], "wrong": []})
            fh.write(_SECTION + "\n")
            fh.write(f"Class: {cls}\n")
            fh.write(_SECTION + "\n")
            for tag, indices in [
                ("CORRECT", buckets["correct"]),
                ("WRONG", buckets["wrong"]),
            ]:
                fh.write(_RULE + "\n")
                fh.write(f"{tag} samples\n")
                fh.write(_RULE + "\n")
                if not indices:
                    fh.write("(none)\n")
                    continue
                for i in indices[:per_bucket]:
                    n_tok = int(len(test_tokens[i]))
                    tissue = str(test_tissues[i])
                    pred = str(predictions[i])
                    fh.write(
                        f"  idx={i}  n_tokens={n_tok:>4d}  "
                        f"tissue={tissue[:24]:<24s}  pred={pred[:36]}\n"
                    )
            fh.write("\n")
