"""Per-variant prediction dumps for the gnomAD AF-correlation evaluator.

This mirrors :mod:`molcrawl.tasks.evaluation.clinvar.predictions_log`
but the target is continuous (allele frequency) rather than a binary
pathogenicity label, so the narrative samples rows by AF bin (rare vs
common) and reports the observed score rank alongside the AF rank so
readers can see whether the model's ordering aligns with allele
frequency at all.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .splits import DEFAULT_AF_BINS, assign_af_bins, bin_label

logger = logging.getLogger(__name__)


def write_predictions(
    output_dir: Path,
    dataset: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    score_distribution: Optional[Dict[str, Any]] = None,
    sampling: Optional[Dict[str, Any]] = None,
    arch: Optional[str] = None,
    modality: Optional[str] = None,
    preview_count: int = 20,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = dataset.reset_index(drop=True).copy()
    scores = np.asarray(predictions["scores"], dtype=float)
    ref_ll = np.asarray(predictions["reference_log_likelihood"], dtype=float)
    var_ll = np.asarray(predictions["variant_log_likelihood"], dtype=float)
    af = df["allele_frequency"].to_numpy(dtype=float)

    # Rank ordering: AF rank and score rank, used to visualise per-row
    # contribution to Spearman. Rank 1 = smallest value.
    af_rank = pd.Series(af).rank(method="average").to_numpy()
    score_rank = pd.Series(scores).rank(method="average").to_numpy()

    jsonl_path = output_dir / "predictions.jsonl"
    _write_jsonl(
        jsonl_path, df, ref_ll, var_ll, scores, af, af_rank, score_rank
    )

    txt_path = output_dir / "predictions.txt"
    _write_narrative(
        txt_path,
        df,
        ref_ll=ref_ll,
        var_ll=var_ll,
        scores=scores,
        af=af,
        af_rank=af_rank,
        score_rank=score_rank,
        score_distribution=score_distribution,
        sampling=sampling,
        arch=arch,
        modality=modality,
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
    df: pd.DataFrame,
    ref_ll: np.ndarray,
    var_ll: np.ndarray,
    scores: np.ndarray,
    af: np.ndarray,
    af_rank: np.ndarray,
    score_rank: np.ndarray,
) -> None:
    def _value(col: str, row: pd.Series) -> Any:
        if col not in row:
            return None
        v = row[col]
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    with path.open("w", encoding="utf-8") as fh:
        for i, row in df.iterrows():
            record: Dict[str, Any] = {
                "index": int(i),
                "chrom": _stringify(_value("chrom", row)),
                "pos": _maybe_int(_value("pos", row)),
                "ref_allele": _stringify(_value("ref", row)),
                "alt_allele": _stringify(_value("alt", row)),
                "reference_sequence": _stringify(_value("reference_sequence", row)),
                "variant_sequence": _stringify(_value("variant_sequence", row)),
                "allele_frequency": float(af[i]),
                "af_rank": float(af_rank[i]),
                "ll_ref": float(ref_ll[i]),
                "ll_var": float(var_ll[i]),
                "score": float(scores[i]),
                "score_rank": float(score_rank[i]),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    df: pd.DataFrame,
    ref_ll: np.ndarray,
    var_ll: np.ndarray,
    scores: np.ndarray,
    af: np.ndarray,
    af_rank: np.ndarray,
    score_rank: np.ndarray,
    score_distribution: Optional[Dict[str, Any]],
    sampling: Optional[Dict[str, Any]],
    arch: Optional[str],
    modality: Optional[str],
    preview_count: int,
) -> None:
    preview_indices = _pick_preview_indices(af, preview_count)

    with path.open("w", encoding="utf-8") as fh:
        _write_preamble(fh, df, af, sampling, score_distribution, arch, modality)
        for rank, i in enumerate(preview_indices, start=1):
            _write_one_example(
                fh,
                rank=rank,
                total_preview=len(preview_indices),
                row=df.iloc[i],
                row_index=int(i),
                ll_ref=float(ref_ll[i]),
                ll_var=float(var_ll[i]),
                score=float(scores[i]),
                af=float(af[i]),
                af_rank=float(af_rank[i]),
                score_rank=float(score_rank[i]),
                n_total=len(df),
            )


def _write_preamble(
    fh,
    df: pd.DataFrame,
    af: np.ndarray,
    sampling: Optional[Dict[str, Any]],
    score_distribution: Optional[Dict[str, Any]],
    arch: Optional[str],
    modality: Optional[str],
) -> None:
    fh.write("gnomAD AF-correlation per-variant prediction narrative\n")
    fh.write(_SECTION + "\n")
    fh.write(f"arch            : {arch or '?'}\n")
    fh.write(f"modality        : {modality or '?'}\n")
    fh.write(f"n_total         : {len(df)}\n")
    if len(af):
        fh.write(
            f"AF range        : min={af.min():.2e}  max={af.max():.2e}  "
            f"median={float(np.median(af)):.2e}\n"
        )

    if sampling:
        fh.write(
            "sampling        : n_per_bin={n_per_bin} seed={seed}\n".format(
                n_per_bin=sampling.get("n_per_bin"),
                seed=sampling.get("seed"),
            )
        )

    bin_ids = assign_af_bins(af)
    fh.write("\n-- AF-bin counts --\n")
    for i, (lo, hi) in enumerate(DEFAULT_AF_BINS):
        n = int((bin_ids == i).sum())
        fh.write(f"  {bin_label(lo, hi):>12s} : {n:>5d}\n")

    if score_distribution:
        fh.write("\n-- per-bin LL / score statistics --\n")
        for label, d in score_distribution.items():
            if not d or d.get("n", 0) == 0:
                continue
            fh.write(
                f"  {label:>12s}  n={d['n']:>5d}  "
                f"LL(ref)={d['ll_ref_mean']:+.4f} ± {d['ll_ref_std']:.4f}  "
                f"LL(var)={d['ll_var_mean']:+.4f} ± {d['ll_var_std']:.4f}  "
                f"score={d['score_mean']:+.4f} ± {d['score_std']:.4f}\n"
            )

    fh.write("\n")
    fh.write(
        "Scoring rule: score = LL(var) − LL(ref). Higher score means the model "
        "considers the variant sequence more likely than the reference, which "
        "is the expected direction for common alleles (the model should treat "
        "frequent alleles as 'natural'). A positive Spearman(AF, score) means "
        "the ordering agrees with allele frequency.\n"
    )
    fh.write(_SECTION + "\n\n")


def _pick_preview_indices(af: np.ndarray, preview_count: int) -> List[int]:
    """Return indices sampled from top / bottom / middle AF strata.

    Seeing the extremes is where signal should be most visible; the
    middle rows give a feel for "typical" behaviour.
    """
    if preview_count <= 0 or len(af) == 0:
        return []
    preview_count = min(preview_count, len(af))

    rng = np.random.default_rng(seed=0)

    order = np.argsort(af)
    n = len(af)
    top_k = max(1, preview_count // 3)
    bot_k = max(1, preview_count // 3)
    mid_k = preview_count - top_k - bot_k

    bottom = order[:bot_k]
    top = order[-top_k:]
    mid_pool = order[n // 4 : 3 * n // 4]
    mid = rng.choice(mid_pool, size=min(mid_k, len(mid_pool)), replace=False) if len(mid_pool) else np.array([], dtype=int)

    picked = np.concatenate([bottom, mid, top])
    return sorted(set(int(x) for x in picked))


def _write_one_example(
    fh,
    rank: int,
    total_preview: int,
    row: pd.Series,
    row_index: int,
    ll_ref: float,
    ll_var: float,
    score: float,
    af: float,
    af_rank: float,
    score_rank: float,
    n_total: int,
) -> None:
    chrom = row.get("chrom", "?")
    pos = row.get("pos", "?")
    ref_allele = row.get("ref", "?")
    alt_allele = row.get("alt", "?")

    ref_seq = str(row.get("reference_sequence", ""))
    var_seq = str(row.get("variant_sequence", ""))

    fh.write(_SECTION + "\n")
    fh.write(
        f"Example {rank} of {total_preview} (dataset index {row_index})\n"
    )
    fh.write(
        f"Allele freq : {af:.4e}   (AF rank {af_rank:.0f} / {n_total}, "
        f"score rank {score_rank:.0f} / {n_total}, rank gap "
        f"{score_rank - af_rank:+.0f})\n"
    )
    fh.write(
        f"Location    : chr{chrom}:{pos}  {ref_allele}>{alt_allele}\n"
    )
    fh.write(_RULE + "\n")
    fh.write(
        f"LL(ref)     : {ll_ref:+.6f}\n"
        f"LL(var)     : {ll_var:+.6f}\n"
        f"score (Δ)   : {score:+.6f}   (= LL(var) − LL(ref))\n"
    )
    fh.write(_RULE + "\n")
    _render_sequence_pair(fh, ref_seq, var_seq)
    fh.write(_SECTION + "\n\n")


def _render_sequence_pair(fh, ref_seq: str, var_seq: str, wrap: int = 72) -> None:
    if not ref_seq or not var_seq:
        fh.write("(sequences unavailable)\n")
        return
    diff_positions = [
        i for i in range(min(len(ref_seq), len(var_seq))) if ref_seq[i] != var_seq[i]
    ]
    diff_set = set(diff_positions)
    for start in range(0, max(len(ref_seq), len(var_seq)), wrap):
        end = start + wrap
        ref_chunk = ref_seq[start:end]
        var_chunk = var_seq[start:end]
        caret = "".join(
            "^" if (start + i) in diff_set else " " for i in range(len(ref_chunk))
        )
        fh.write(f"  ref [{start:>4d}] {ref_chunk}\n")
        fh.write(f"  var [{start:>4d}] {var_chunk}\n")
        if any(c == "^" for c in caret):
            fh.write(f"             {caret}\n")
    if diff_positions:
        summary = ", ".join(
            f"pos {p} ({ref_seq[p]}→{var_seq[p]})" for p in diff_positions[:5]
        )
        if len(diff_positions) > 5:
            summary += f", +{len(diff_positions) - 5} more"
        fh.write(f"  differences: {summary}\n")
