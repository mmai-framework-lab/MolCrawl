"""Per-perturbation predictions dump.

Two artefacts:

- ``predictions.jsonl`` — one record per held-out perturbation: target
  gene, observed-vs-predicted per-gene Spearman / Pearson, top-error
  genes (largest |observed - predicted|).
- ``predictions.txt`` — narrative preview. Samples best-fit (highest
  Spearman) and worst-fit (lowest Spearman) perturbations so the
  reader can see which KO targets the encoder learned vs. fumbled.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation._base import default_registry

logger = logging.getLogger(__name__)


def write_predictions(
    output_dir: Path,
    test_df: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    arch: Optional[str] = None,
    preview_count: int = 16,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    txt_path = output_dir / "predictions.txt"

    per_pert = _score_per_pert(test_df, observed, predicted)
    _write_jsonl(jsonl_path, per_pert)
    _write_narrative(
        txt_path,
        per_pert=per_pert,
        arch=arch,
        n_genes=int(observed.shape[1]) if observed.size else 0,
        preview_count=preview_count,
    )

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


def _score_per_pert(
    test_df: pd.DataFrame, observed: np.ndarray, predicted: np.ndarray
) -> list:
    rows = []
    perturbations = test_df["perturbation"].astype(str).tolist()
    for i, pert in enumerate(perturbations):
        if i >= observed.shape[0] or i >= predicted.shape[0]:
            break
        obs = observed[i]
        pred = predicted[i]
        rec: Dict[str, Any] = {
            "index": int(i),
            "perturbation": pert,
            "n_genes": int(obs.size),
            "observed_norm": float(np.linalg.norm(obs)),
            "predicted_norm": float(np.linalg.norm(pred)),
        }
        if obs.size >= 2 and np.std(obs) > 0 and np.std(pred) > 0:
            rec["spearman"] = float(default_registry.compute("spearman", obs, pred))
            rec["pearson"] = float(default_registry.compute("pearson", obs, pred))
        else:
            rec["spearman"] = None
            rec["pearson"] = None
        # Top-3 absolute-error genes
        err = np.abs(obs - pred)
        if err.size:
            top_k = np.argsort(err)[-3:][::-1]
            rec["top_error_gene_idx"] = [int(j) for j in top_k]
            rec["top_error_value"] = [float(err[j]) for j in top_k]
        rows.append(rec)
    return rows


def _write_jsonl(path: Path, per_pert: list) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in per_pert:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    per_pert: list,
    arch: Optional[str],
    n_genes: int,
    preview_count: int,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Replogle Perturb-seq per-perturbation narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch       : {arch or '?'}\n")
        fh.write(f"n_test     : {len(per_pert)}\n")
        fh.write(f"n_genes    : {n_genes}\n")

        spear = [
            r["spearman"] for r in per_pert if r.get("spearman") is not None
        ]
        if spear:
            fh.write(
                f"spearman   : mean={float(np.mean(spear)):+.3f} "
                f"min={float(min(spear)):+.3f} max={float(max(spear)):+.3f} "
                f"(scored={len(spear)} of {len(per_pert)})\n"
            )

        fh.write("\n")
        fh.write(
            "Block A samples WORST-FIT perturbations (lowest Spearman — model "
            "predicts the wrong response shape). Block B samples BEST-FIT "
            "perturbations (highest Spearman — encoder captured the response).\n"
        )
        fh.write(_SECTION + "\n\n")

        scored = [r for r in per_pert if r.get("spearman") is not None]
        if not scored:
            fh.write("(no scored perturbations — every test row had constant obs/pred)\n")
            return
        scored.sort(key=lambda r: r["spearman"])
        half = max(1, preview_count // 2)
        bot = scored[:half]
        top = list(reversed(scored[-half:]))

        for label, rows in [("WORST-FIT", bot), ("BEST-FIT", top)]:
            fh.write(_SECTION + "\n")
            fh.write(f"{label} samples\n")
            fh.write(_SECTION + "\n")
            for r in rows:
                fh.write(_RULE + "\n")
                sp = r["spearman"]
                pe = r["pearson"]
                fh.write(
                    f"  perturbation : {r['perturbation']}\n"
                    f"  spearman     : {sp:+.3f}  pearson: {pe:+.3f}\n"
                    f"  obs_norm     : {r['observed_norm']:.3f}  "
                    f"pred_norm: {r['predicted_norm']:.3f}\n"
                )
                if "top_error_gene_idx" in r:
                    pairs = ", ".join(
                        f"gene_idx={i} err={v:.2f}"
                        for i, v in zip(r["top_error_gene_idx"], r["top_error_value"])
                    )
                    fh.write(f"  top errors   : {pairs}\n")
            fh.write("\n")
