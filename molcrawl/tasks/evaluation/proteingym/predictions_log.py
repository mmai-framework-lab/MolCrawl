"""Per-variant prediction dumps for the ProteinGym evaluator.

Mirrors the ClinVar / gnomAD narrative layers but the target is the
continuous experimental DMS score rather than a binary label, so the
narrative samples rows by DMS-score quantile (low / mid / high) so
reviewers can see whether the model score tracks fitness order.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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
    ref_ll = np.asarray(predictions["reference_ll"], dtype=float)
    mut_ll = np.asarray(predictions["mutated_ll"], dtype=float)
    dms = df["DMS_score"].to_numpy(dtype=float)

    dms_rank = pd.Series(dms).rank(method="average").to_numpy()
    score_rank = pd.Series(scores).rank(method="average").to_numpy()

    jsonl_path = output_dir / "predictions.jsonl"
    _write_jsonl(
        jsonl_path, df, ref_ll, mut_ll, scores, dms, dms_rank, score_rank
    )

    txt_path = output_dir / "predictions.txt"
    _write_narrative(
        txt_path,
        df,
        ref_ll=ref_ll,
        mut_ll=mut_ll,
        scores=scores,
        dms=dms,
        dms_rank=dms_rank,
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


def _write_jsonl(
    path: Path,
    df: pd.DataFrame,
    ref_ll: np.ndarray,
    mut_ll: np.ndarray,
    scores: np.ndarray,
    dms: np.ndarray,
    dms_rank: np.ndarray,
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
                "mutant": _stringify(_value("mutant", row)),
                "mutated_sequence": _stringify(_value("mutated_sequence", row)),
                "wildtype_sequence": _stringify(_value("wildtype_sequence", row)),
                "dms_score": float(dms[i]),
                "dms_bin_score": (
                    int(row["DMS_bin_score"])
                    if "DMS_bin_score" in row
                    and not pd.isna(row.get("DMS_bin_score"))
                    else None
                ),
                "dms_rank": float(dms_rank[i]),
                "ll_ref": float(ref_ll[i]),
                "ll_mut": float(mut_ll[i]),
                "score": float(scores[i]),
                "score_rank": float(score_rank[i]),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    df: pd.DataFrame,
    ref_ll: np.ndarray,
    mut_ll: np.ndarray,
    scores: np.ndarray,
    dms: np.ndarray,
    dms_rank: np.ndarray,
    score_rank: np.ndarray,
    score_distribution: Optional[Dict[str, Any]],
    sampling: Optional[Dict[str, Any]],
    arch: Optional[str],
    modality: Optional[str],
    preview_count: int,
) -> None:
    preview_indices = _pick_preview_indices(dms, preview_count)

    with path.open("w", encoding="utf-8") as fh:
        _write_preamble(fh, df, dms, sampling, score_distribution, arch, modality)
        for rank, i in enumerate(preview_indices, start=1):
            _write_one_example(
                fh,
                rank=rank,
                total_preview=len(preview_indices),
                row=df.iloc[i],
                row_index=int(i),
                ll_ref=float(ref_ll[i]),
                ll_mut=float(mut_ll[i]),
                score=float(scores[i]),
                dms=float(dms[i]),
                dms_rank=float(dms_rank[i]),
                score_rank=float(score_rank[i]),
                n_total=len(df),
            )


def _write_preamble(
    fh,
    df: pd.DataFrame,
    dms: np.ndarray,
    sampling: Optional[Dict[str, Any]],
    score_distribution: Optional[Dict[str, Any]],
    arch: Optional[str],
    modality: Optional[str],
) -> None:
    fh.write("ProteinGym per-variant prediction narrative\n")
    fh.write(_SECTION + "\n")
    fh.write(f"arch            : {arch or '?'}\n")
    fh.write(f"modality        : {modality or '?'}\n")
    fh.write(f"n_total         : {len(df)}\n")
    if len(dms):
        fh.write(
            f"DMS range       : min={dms.min():+.3f}  max={dms.max():+.3f}  "
            f"median={float(np.median(dms)):+.3f}\n"
        )
    if "DMS_bin_score" in df.columns:
        vc = df["DMS_bin_score"].value_counts(dropna=False).to_dict()
        fh.write(f"DMS_bin dist    : {vc}\n")
    if sampling:
        fh.write(
            "sampling        : n_examples={n_examples} stratify_bin={stratify_bin} seed={seed}\n".format(
                n_examples=sampling.get("n_examples"),
                stratify_bin=sampling.get("stratify_bin"),
                seed=sampling.get("seed"),
            )
        )

    if score_distribution:
        fh.write("\n-- per-class LL / score statistics --\n")
        for label, d in score_distribution.items():
            if not d or d.get("n", 0) == 0:
                continue
            fh.write(
                f"  {label:>16s}  n={d['n']:>5d}  "
                f"DMS={d.get('dms_mean', 0.0):+.3f} ± {d.get('dms_std', 0.0):.3f}  "
                f"LL(ref)={d.get('ll_ref_mean', 0.0):+.3f}  "
                f"LL(mut)={d.get('ll_mut_mean', 0.0):+.3f}  "
                f"score={d.get('score_mean', 0.0):+.4f} ± {d.get('score_std', 0.0):.4f}\n"
            )

    fh.write("\n")
    fh.write(
        "Scoring rule: score = LL(mut) − LL(ref). Higher score means the model "
        "considers the variant sequence more likely than the wildtype. A positive "
        "Spearman(DMS_score, score) means the model's ranking agrees with the "
        "experimental fitness: more-fit (higher DMS) variants get larger scores.\n"
    )
    fh.write(_SECTION + "\n\n")


def _pick_preview_indices(dms: np.ndarray, preview_count: int) -> List[int]:
    if preview_count <= 0 or len(dms) == 0:
        return []
    preview_count = min(preview_count, len(dms))

    rng = np.random.default_rng(seed=0)
    order = np.argsort(dms)
    n = len(dms)
    top_k = max(1, preview_count // 3)
    bot_k = max(1, preview_count // 3)
    mid_k = preview_count - top_k - bot_k

    bottom = order[:bot_k]
    top = order[-top_k:]
    mid_pool = order[n // 4 : 3 * n // 4]
    mid = (
        rng.choice(mid_pool, size=min(mid_k, len(mid_pool)), replace=False)
        if len(mid_pool)
        else np.array([], dtype=int)
    )

    picked = np.concatenate([bottom, mid, top])
    return sorted(set(int(x) for x in picked))


def _write_one_example(
    fh,
    rank: int,
    total_preview: int,
    row: pd.Series,
    row_index: int,
    ll_ref: float,
    ll_mut: float,
    score: float,
    dms: float,
    dms_rank: float,
    score_rank: float,
    n_total: int,
) -> None:
    mutant = row.get("mutant", "?")
    dms_bin = row.get("DMS_bin_score", None)
    dms_bin_name = (
        "functional" if dms_bin == 1
        else "non-functional" if dms_bin == 0
        else "?"
    )

    mut_seq = str(row.get("mutated_sequence", ""))
    ref_seq = str(row.get("wildtype_sequence", ""))

    fh.write(_SECTION + "\n")
    fh.write(
        f"Example {rank} of {total_preview} (dataset index {row_index})\n"
    )
    fh.write(
        f"Mutant      : {mutant}   DMS={dms:+.3f}  bin={dms_bin_name}\n"
        f"Ranks       : DMS rank {dms_rank:.0f} / {n_total}, "
        f"score rank {score_rank:.0f} / {n_total}, "
        f"rank gap {score_rank - dms_rank:+.0f}\n"
    )
    fh.write(_RULE + "\n")
    fh.write(
        f"LL(ref)     : {ll_ref:+.6f}\n"
        f"LL(mut)     : {ll_mut:+.6f}\n"
        f"score (Δ)   : {score:+.6f}   (= LL(mut) − LL(ref))\n"
    )
    fh.write(_RULE + "\n")
    _render_sequence_pair(fh, ref_seq, mut_seq)
    fh.write(_SECTION + "\n\n")


def _render_sequence_pair(fh, ref_seq: str, mut_seq: str, wrap: int = 72) -> None:
    if not ref_seq or not mut_seq:
        fh.write("(sequences unavailable)\n")
        return
    diff_positions = [
        i for i in range(min(len(ref_seq), len(mut_seq))) if ref_seq[i] != mut_seq[i]
    ]
    if not diff_positions:
        # Identity (possible for pure-WT control rows); just print first 72 aa.
        fh.write(f"  ref [   0] {ref_seq[:wrap]}\n")
        fh.write("  (no differences — this row is the wildtype itself)\n")
        return

    # Center the viewing window around the first differing position.
    center = diff_positions[0]
    half = wrap
    start = max(0, center - half // 2)
    end = min(max(len(ref_seq), len(mut_seq)), start + wrap)
    ref_chunk = ref_seq[start:end]
    mut_chunk = mut_seq[start:end]
    caret = "".join(
        "^" if (start + i) in set(diff_positions) else " "
        for i in range(len(ref_chunk))
    )
    fh.write(f"  ref [{start:>4d}] {ref_chunk}\n")
    fh.write(f"  mut [{start:>4d}] {mut_chunk}\n")
    fh.write(f"             {caret}\n")

    summary = ", ".join(
        f"pos {p + 1} ({ref_seq[p]}→{mut_seq[p]})"
        for p in diff_positions[:5]
    )
    if len(diff_positions) > 5:
        summary += f", +{len(diff_positions) - 5} more"
    fh.write(f"  differences: {summary}\n")
