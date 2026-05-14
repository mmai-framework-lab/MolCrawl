"""Per-prompt predictions dump for the ChemLLMBench evaluator.

Two artefacts:

- ``predictions.jsonl`` — one record per example: prompt, gold answer,
  generated prediction, metadata, and (when applicable) per-row
  exact-match / abs-error.
- ``predictions.txt`` — narrative preview. Samples a few correct + a few
  wrong rows so the reader can see the kinds of mistakes the model
  makes (truncated, off-by-one SMILES, hallucinated text, etc.).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def write_predictions(
    output_dir: Path,
    examples: Sequence[Any],
    predictions: Sequence[str],
    task: str,
    task_type: str,
    arch: Optional[str] = None,
    preview_count: int = 20,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    _write_jsonl(jsonl_path, examples, predictions, task, task_type)
    _write_narrative(
        txt_path,
        examples=examples,
        predictions=predictions,
        task=task,
        task_type=task_type,
        arch=arch,
        preview_count=preview_count,
    )

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


def _is_correct(pred: str, ref: str, task_type: str) -> Optional[bool]:
    if task_type in ("exact", "smiles"):
        return str(pred).strip() == str(ref).strip()
    return None  # text / regression — no clean per-row correct flag


def _write_jsonl(
    path: Path,
    examples: Sequence[Any],
    predictions: Sequence[str],
    task: str,
    task_type: str,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, ex in enumerate(examples):
            pred = predictions[i] if i < len(predictions) else ""
            rec: Dict[str, Any] = {
                "index": int(i),
                "task": task,
                "task_type": task_type,
                "prompt": str(ex.prompt),
                "answer": str(ex.answer),
                "prediction": str(pred),
                "metadata": dict(ex.metadata or {}),
            }
            correct = _is_correct(pred, ex.answer, task_type)
            if correct is not None:
                rec["exact_match"] = bool(correct)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    examples: Sequence[Any],
    predictions: Sequence[str],
    task: str,
    task_type: str,
    arch: Optional[str],
    preview_count: int,
) -> None:
    n = len(examples)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"ChemLLMBench {task} per-prompt narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch       : {arch or '?'}\n")
        fh.write(f"task       : {task}\n")
        fh.write(f"task_type  : {task_type}\n")
        fh.write(f"n_examples : {n}\n")

        correct_idx: List[int] = []
        wrong_idx: List[int] = []
        for i, ex in enumerate(examples):
            pred = predictions[i] if i < len(predictions) else ""
            c = _is_correct(pred, ex.answer, task_type)
            if c is True:
                correct_idx.append(i)
            elif c is False:
                wrong_idx.append(i)

        if task_type in ("exact", "smiles") and n:
            fh.write(
                f"correct    : {len(correct_idx)} / {n} "
                f"(={len(correct_idx) / n:.3f})\n"
            )
        fh.write("\n")
        if task_type in ("exact", "smiles"):
            fh.write(
                "Block A samples CORRECT predictions; Block B samples WRONG ones "
                "so the reader can see typical failure modes (off-by-one SMILES, "
                "truncated output, hallucinated tokens, etc.).\n"
            )
        else:
            fh.write(
                "Generative tasks (text / regression) don't carry a clean per-row "
                "correct flag; the block below samples the first preview_count rows "
                "verbatim so the reader can sanity-check generation quality.\n"
            )
        fh.write(_SECTION + "\n\n")

        half = max(1, preview_count // 2)
        if task_type in ("exact", "smiles"):
            for label, pool in [
                ("CORRECT", correct_idx),
                ("WRONG", wrong_idx),
            ]:
                fh.write(_SECTION + "\n")
                fh.write(f"{label} samples\n")
                fh.write(_SECTION + "\n")
                if not pool:
                    fh.write("(none)\n\n")
                    continue
                for i in pool[:half]:
                    _write_one(fh, i, examples[i], predictions[i])
                fh.write("\n")
        else:
            fh.write(_SECTION + "\n")
            fh.write("First-N samples\n")
            fh.write(_SECTION + "\n")
            for i in range(min(preview_count, n)):
                _write_one(fh, i, examples[i], predictions[i])


def _write_one(fh, i: int, ex: Any, pred: str) -> None:
    fh.write(_RULE + "\n")
    fh.write(f"Example {i}\n")
    prompt = str(ex.prompt)
    answer = str(ex.answer)
    pred_s = str(pred)
    fh.write(f"prompt     : {prompt[:300]}{'...' if len(prompt) > 300 else ''}\n")
    fh.write(f"gold       : {answer[:300]}{'...' if len(answer) > 300 else ''}\n")
    fh.write(f"prediction : {pred_s[:300]}{'...' if len(pred_s) > 300 else ''}\n")
