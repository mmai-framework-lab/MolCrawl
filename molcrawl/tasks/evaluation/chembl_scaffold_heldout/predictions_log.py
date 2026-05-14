"""Per-row predictions dump for the ChEMBL scaffold held-out evaluator.

Two artefacts:

- ``predictions.jsonl`` — one row per held-out SMILES with per-token mean
  log-likelihood (perplexity mode) or probe score + label (probe mode).
- ``predictions.txt`` — narrative preview. Perplexity mode samples
  best-fit (highest LL) and worst-fit (lowest LL) SMILES so the reader
  can see *what kind* of molecule the model finds easy / hard. Probe
  mode samples top- and bottom-scored molecules per class.
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
    smiles: Sequence[str],
    mode: str,
    log_likelihood: Optional[Sequence[float]] = None,
    probe_scores: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[Any]] = None,
    label_column: Optional[str] = None,
    arch: Optional[str] = None,
    preview_count: int = 30,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    _write_jsonl(
        jsonl_path,
        smiles=smiles,
        mode=mode,
        log_likelihood=log_likelihood,
        probe_scores=probe_scores,
        labels=labels,
        label_column=label_column,
    )
    _write_narrative(
        txt_path,
        smiles=smiles,
        mode=mode,
        log_likelihood=log_likelihood,
        probe_scores=probe_scores,
        labels=labels,
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
    smiles: Sequence[str],
    mode: str,
    log_likelihood: Optional[Sequence[float]],
    probe_scores: Optional[Sequence[float]],
    labels: Optional[Sequence[Any]],
    label_column: Optional[str],
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, smi in enumerate(smiles):
            record: Dict[str, Any] = {
                "index": int(i),
                "smiles": str(smi),
                "smiles_length": int(len(smi)),
                "mode": mode,
            }
            if log_likelihood is not None and i < len(log_likelihood):
                ll = float(log_likelihood[i])
                record["log_likelihood"] = None if math.isnan(ll) else ll
                record["perplexity"] = (
                    None if math.isnan(ll) else math.exp(-ll)
                )
            if probe_scores is not None and i < len(probe_scores):
                ps = float(probe_scores[i])
                record["probe_score"] = None if math.isnan(ps) else ps
            if labels is not None and i < len(labels):
                lbl = labels[i]
                if lbl is None or (isinstance(lbl, float) and math.isnan(lbl)):
                    record[f"label.{label_column or 'y'}"] = None
                else:
                    record[f"label.{label_column or 'y'}"] = (
                        int(lbl) if isinstance(lbl, (bool, np.integer, int)) else float(lbl)
                    )
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    smiles: Sequence[str],
    mode: str,
    log_likelihood: Optional[Sequence[float]],
    probe_scores: Optional[Sequence[float]],
    labels: Optional[Sequence[Any]],
    label_column: Optional[str],
    arch: Optional[str],
    preview_count: int,
) -> None:
    n = len(smiles)
    lengths = np.array([len(s) for s in smiles], dtype=float)

    with path.open("w", encoding="utf-8") as fh:
        fh.write("ChEMBL scaffold held-out per-row narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch         : {arch or '?'}\n")
        fh.write(f"mode         : {mode}\n")
        fh.write(f"n_heldout    : {n}\n")
        if lengths.size:
            fh.write(
                f"smiles len   : mean={lengths.mean():.1f} std={lengths.std():.1f} "
                f"min={lengths.min():.0f} median={np.median(lengths):.0f} "
                f"max={lengths.max():.0f}\n"
            )

        if mode == "perplexity" and log_likelihood is not None:
            ll = np.asarray(log_likelihood, dtype=float)
            valid = ll[~np.isnan(ll)]
            if valid.size:
                fh.write(
                    f"mean LL      : {valid.mean():+.4f}  "
                    f"perplexity={math.exp(-float(valid.mean())):.3f}\n"
                )
                fh.write(
                    f"LL spread    : min={valid.min():+.4f} median={np.median(valid):+.4f} "
                    f"max={valid.max():+.4f}\n"
                )
            fh.write("\n")
            fh.write(
                "The two blocks below sample (A) the SMILES the model fits "
                "BEST (highest mean log-likelihood — typical chemical motifs) "
                "and (B) the SMILES it fits WORST (lowest LL — structurally "
                "unusual scaffolds, the genuine OOD signal).\n"
            )
            fh.write(_SECTION + "\n\n")
            _perplexity_preview(
                fh,
                smiles=smiles,
                log_likelihood=log_likelihood,
                preview_count=preview_count,
            )
        elif mode == "probe" and probe_scores is not None:
            if labels is not None:
                lbl_arr = pd.Series(labels).astype(float)
                vc = lbl_arr.dropna().astype(int).value_counts().sort_index().to_dict()
                fh.write(f"label dist   : {vc}\n")
            fh.write("\n")
            fh.write(
                "Blocks below sample the highest- and lowest-probe-scored "
                "molecules per class so the reader can sanity-check what the "
                "linear probe is keying on.\n"
            )
            fh.write(_SECTION + "\n\n")
            _probe_preview(
                fh,
                smiles=smiles,
                probe_scores=probe_scores,
                labels=labels,
                label_column=label_column,
                preview_count=preview_count,
            )
        else:
            fh.write("(no scoring data available for narrative preview)\n")


def _perplexity_preview(
    fh,
    smiles: Sequence[str],
    log_likelihood: Sequence[float],
    preview_count: int,
) -> None:
    ll = np.asarray(log_likelihood, dtype=float)
    valid_idx = np.where(~np.isnan(ll))[0]
    if valid_idx.size == 0:
        fh.write("(no valid log-likelihood values)\n")
        return
    order = valid_idx[np.argsort(ll[valid_idx])]
    half = max(1, preview_count // 2)
    bot = list(order[:half])           # lowest LL = worst-fit / hardest
    top = list(order[-half:][::-1])    # highest LL = best-fit / typical

    rank = 0
    for label, indices in [
        ("WORST-FIT (lowest LL — model surprised)", bot),
        ("BEST-FIT (highest LL — model recognises)", top),
    ]:
        fh.write(_SECTION + "\n")
        fh.write(f"{label}\n")
        fh.write(_SECTION + "\n")
        for i in indices:
            rank += 1
            v = float(ll[i])
            fh.write(_RULE + "\n")
            fh.write(
                f"Example {rank}  index={i}  length={len(smiles[i])}  "
                f"LL={v:+.4f}  perplexity={math.exp(-v):.3f}\n"
            )
            fh.write(f"SMILES   : {smiles[i]}\n")
        fh.write("\n")


def _probe_preview(
    fh,
    smiles: Sequence[str],
    probe_scores: Sequence[float],
    labels: Optional[Sequence[Any]],
    label_column: Optional[str],
    preview_count: int,
) -> None:
    scores = np.asarray(probe_scores, dtype=float)
    valid_mask = ~np.isnan(scores)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        fh.write("(no valid probe scores)\n")
        return
    order = valid_idx[np.argsort(scores[valid_idx])]
    half = max(1, preview_count // 2)
    bot = list(order[:half])
    top = list(order[-half:][::-1])

    label_col = label_column or "y"
    rank = 0
    for tag, indices in [
        ("LOWEST probe score", bot),
        ("HIGHEST probe score", top),
    ]:
        fh.write(_SECTION + "\n")
        fh.write(f"{tag}\n")
        fh.write(_SECTION + "\n")
        for i in indices:
            rank += 1
            sc = float(scores[i])
            lbl = labels[i] if labels is not None else None
            lbl_str = (
                "?"
                if lbl is None or (isinstance(lbl, float) and math.isnan(lbl))
                else f"{int(lbl)}"
            )
            fh.write(_RULE + "\n")
            fh.write(
                f"Example {rank}  index={i}  length={len(smiles[i])}  "
                f"score={sc:+.4f}  label.{label_col}={lbl_str}\n"
            )
            fh.write(f"SMILES   : {smiles[i]}\n")
        fh.write("\n")
