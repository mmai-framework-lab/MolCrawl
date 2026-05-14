"""Per-pair predictions dump for the molecule_nat_lang evaluator."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def write_predictions(
    output_dir: Path,
    pairs: pd.DataFrame,
    log_likelihoods: Sequence[float],
    smiles_column: str,
    caption_column: str,
    template: str,
    arch: Optional[str] = None,
    preview_count: int = 20,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    _write_jsonl(jsonl_path, pairs, log_likelihoods, smiles_column, caption_column, template)
    _write_narrative(
        txt_path,
        pairs=pairs,
        log_likelihoods=log_likelihoods,
        smiles_column=smiles_column,
        caption_column=caption_column,
        arch=arch,
        preview_count=preview_count,
    )

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


def _write_jsonl(
    path: Path,
    pairs: pd.DataFrame,
    log_likelihoods: Sequence[float],
    smiles_column: str,
    caption_column: str,
    template: str,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, row in pairs.reset_index(drop=True).iterrows():
            ll = (
                float(log_likelihoods[i])
                if i < len(log_likelihoods) and log_likelihoods[i] is not None
                else float("nan")
            )
            rec: Dict[str, Any] = {
                "index": int(i),
                "smiles": str(row.get(smiles_column, "")),
                "caption": str(row.get(caption_column, "")),
                "template": template,
                "log_likelihood": None if math.isnan(ll) else ll,
                "perplexity": None if math.isnan(ll) else math.exp(-ll),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    pairs: pd.DataFrame,
    log_likelihoods: Sequence[float],
    smiles_column: str,
    caption_column: str,
    arch: Optional[str],
    preview_count: int,
) -> None:
    ll = np.asarray(log_likelihoods, dtype=float)
    valid_idx = np.where(~np.isnan(ll))[0]

    with path.open("w", encoding="utf-8") as fh:
        fh.write("molecule_nat_lang per-pair narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch         : {arch or '?'}\n")
        fh.write(f"n_pairs      : {len(pairs)}\n")
        fh.write(f"n_scored     : {valid_idx.size}\n")
        if valid_idx.size:
            fh.write(
                f"mean LL      : {ll[valid_idx].mean():+.4f}  "
                f"perplexity={math.exp(-float(ll[valid_idx].mean())):.3f}\n"
            )
            fh.write(
                f"LL spread    : min={ll[valid_idx].min():+.4f} "
                f"median={np.median(ll[valid_idx]):+.4f} "
                f"max={ll[valid_idx].max():+.4f}\n"
            )
        fh.write("\n")
        fh.write(
            "Blocks below show pairs the model fits BEST (highest mean log-likelihood, "
            "i.e. caption+SMILES the model reproduces easily) and WORST (the most "
            "surprising pairs — usually long SELFIES with terse captions or rare "
            "chemical entities).\n"
        )
        fh.write(_SECTION + "\n\n")

        if valid_idx.size == 0:
            fh.write("(no valid scores)\n")
            return

        order = valid_idx[np.argsort(ll[valid_idx])]
        half = max(1, preview_count // 2)
        bot = list(order[:half])
        top = list(order[-half:][::-1])

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
                row = pairs.iloc[i]
                smi = str(row[smiles_column])
                cap = str(row[caption_column])
                fh.write(_RULE + "\n")
                fh.write(
                    f"Example {rank}  index={i}  "
                    f"LL={v:+.4f}  perplexity={math.exp(-v):.3f}\n"
                )
                fh.write(f"caption   : {cap[:280]}{'...' if len(cap) > 280 else ''}\n")
                fh.write(f"molecule  : {smi[:280]}{'...' if len(smi) > 280 else ''}\n")
            fh.write("\n")
