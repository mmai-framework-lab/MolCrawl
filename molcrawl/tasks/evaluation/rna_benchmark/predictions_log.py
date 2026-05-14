"""Per-cell predictions dump for the rna_benchmark evaluator.

Two artefacts:

- ``predictions.jsonl`` — one record per scored cell: dataset, length,
  per-cell mean log-likelihood, perplexity.
- ``predictions.txt`` — narrative preview. For each tissue, samples the
  cells with the highest and lowest per-cell mean log-likelihood so the
  reader can see *which cells* the model finds easy / hard within a
  given tissue.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def write_predictions(
    output_dir: Path,
    per_cell: Dict[str, Dict[str, Sequence[float]]],
    arch: Optional[str] = None,
    preview_count: int = 6,
) -> Dict[str, str]:
    """Write JSONL + narrative TXT.

    ``per_cell`` is ``{tissue: {"log_likelihood": [...], "token_count": [...]}}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    _write_jsonl(jsonl_path, per_cell)
    _write_narrative(txt_path, per_cell, arch=arch, preview_count=preview_count)

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


def _write_jsonl(path: Path, per_cell: Dict[str, Dict[str, Sequence[float]]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for tissue, data in per_cell.items():
            ll = data.get("log_likelihood", [])
            tc = data.get("token_count", [])
            for i, v in enumerate(ll):
                v_f = float(v) if v is not None else float("nan")
                rec: Dict[str, Any] = {
                    "dataset": tissue,
                    "index_in_group": int(i),
                    "log_likelihood": None if math.isnan(v_f) else v_f,
                    "perplexity": None if math.isnan(v_f) else math.exp(-v_f),
                }
                if i < len(tc):
                    rec["token_count"] = int(tc[i])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    per_cell: Dict[str, Dict[str, Sequence[float]]],
    arch: Optional[str],
    preview_count: int,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("rna_benchmark per-cell narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch         : {arch or '?'}\n")
        fh.write(f"n_groups     : {len(per_cell)}\n")
        total = sum(len(d.get("log_likelihood", [])) for d in per_cell.values())
        fh.write(f"n_cells_total: {total}\n\n")

        fh.write(
            "For each tissue we show the top-K and bottom-K cells by per-cell "
            "mean log-likelihood. Top = the cells the model recognises easily; "
            "bottom = cells the model finds surprising. Big gaps within a tissue "
            "usually signal heterogeneous cell-type composition.\n"
        )
        fh.write(_SECTION + "\n\n")

        half = max(1, preview_count // 2)
        for tissue, data in per_cell.items():
            ll = np.asarray(data.get("log_likelihood", []), dtype=float)
            tc = np.asarray(data.get("token_count", []), dtype=int)
            valid_mask = ~np.isnan(ll)
            valid_idx = np.where(valid_mask)[0]
            if valid_idx.size == 0:
                fh.write(f"--- {tissue}: no valid log-likelihood values\n\n")
                continue
            order = valid_idx[np.argsort(ll[valid_idx])]
            bot = list(order[:half])
            top = list(order[-half:][::-1])

            fh.write(_SECTION + "\n")
            fh.write(
                f"{tissue:<30s}  n={valid_idx.size}  "
                f"mean_ll={ll[valid_idx].mean():+.4f}  "
                f"ppl={math.exp(-float(ll[valid_idx].mean())):.3f}\n"
            )
            fh.write(_SECTION + "\n")
            for label, indices in [
                ("WORST-FIT (lowest LL — model surprised)", bot),
                ("BEST-FIT (highest LL — model recognises)", top),
            ]:
                fh.write(_RULE + "\n")
                fh.write(f"{label}\n")
                fh.write(_RULE + "\n")
                for i in indices:
                    v = float(ll[i])
                    n_tok = int(tc[i]) if i < tc.size else 0
                    fh.write(
                        f"  cell[{i:>4d}]  tokens={n_tok:>4d}  "
                        f"LL={v:+.4f}  ppl={math.exp(-v):.3f}\n"
                    )
            fh.write("\n")
