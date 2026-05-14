"""Per-variant prediction dumps for COSMIC.

Mirrors :mod:`molcrawl.tasks.evaluation.clinvar.predictions_log` but uses
the COSMIC schema names: ``cosmic_label`` (0/1) replaces clinvar's
``pathogenic``, ``FATHMM_PREDICTION`` replaces ``ClinicalSignificance``,
and we surface ``MUTATION_SIGNIFICANCE_TIER`` + ``GENE_NAME`` so reviewers
can see which CMC tier each preview row came from.
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
    threshold: Optional[float],
    score_distribution: Optional[Dict[str, Any]] = None,
    sampling: Optional[Dict[str, Any]] = None,
    arch: Optional[str] = None,
    modality: Optional[str] = None,
    preview_count: int = 20,
) -> Dict[str, str]:
    """Persist per-variant predictions under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = dataset.reset_index(drop=True).copy()
    scores = np.asarray(predictions["scores"], dtype=float)
    ref_ll = np.asarray(predictions["reference_ll"], dtype=float)
    var_ll = np.asarray(predictions["variant_ll"], dtype=float)
    labels = df["cosmic_label"].to_numpy(dtype=int)

    predicted: Optional[np.ndarray] = None
    correct: Optional[np.ndarray] = None
    if threshold is not None:
        predicted = (scores > threshold).astype(int)
        correct = (predicted == labels).astype(int)

    jsonl_path = output_dir / "predictions.jsonl"
    _write_jsonl(jsonl_path, df, ref_ll, var_ll, scores, labels, threshold, predicted, correct)

    txt_path = output_dir / "predictions.txt"
    _write_narrative(
        txt_path,
        df,
        ref_ll,
        var_ll,
        scores,
        labels,
        threshold=threshold,
        predicted=predicted,
        correct=correct,
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
# JSONL dump
# ---------------------------------------------------------------------


def _value(col: str, row: pd.Series) -> Any:
    if col not in row:
        return None
    v = row[col]
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


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


def _write_jsonl(
    path: Path,
    df: pd.DataFrame,
    ref_ll: np.ndarray,
    var_ll: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float],
    predicted: Optional[np.ndarray],
    correct: Optional[np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, row in df.iterrows():
            record: Dict[str, Any] = {
                "index": int(i),
                "gene": _stringify(_value("GENE_NAME", row)),
                "tier": _stringify(_value("MUTATION_SIGNIFICANCE_TIER", row)),
                "chrom": _stringify(_value("chrom", row)),
                "pos": _maybe_int(_value("pos", row)),
                "ref_allele": _stringify(_value("ref", row)),
                "alt_allele": _stringify(_value("alt", row)),
                "reference_sequence": _stringify(_value("reference_sequence", row)),
                "variant_sequence": _stringify(_value("variant_sequence", row)),
                "fathmm_prediction": _stringify(_value("FATHMM_PREDICTION", row)),
                "label_pathogenic": int(labels[i]),
                "ll_ref": float(ref_ll[i]),
                "ll_var": float(var_ll[i]),
                "score": float(scores[i]),
                "threshold": float(threshold) if threshold is not None else None,
                "predicted_pathogenic": (
                    int(predicted[i]) if predicted is not None else None
                ),
                "correct": (
                    bool(correct[i]) if correct is not None else None
                ),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Human-readable narrative
# ---------------------------------------------------------------------


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    df: pd.DataFrame,
    ref_ll: np.ndarray,
    var_ll: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float],
    predicted: Optional[np.ndarray],
    correct: Optional[np.ndarray],
    score_distribution: Optional[Dict[str, Any]],
    sampling: Optional[Dict[str, Any]],
    arch: Optional[str],
    modality: Optional[str],
    preview_count: int,
) -> None:
    preview_indices = _pick_preview_indices(
        labels=labels,
        correct=correct,
        total=len(df),
        preview_count=preview_count,
    )

    with path.open("w", encoding="utf-8") as fh:
        _write_preamble(fh, df, labels, threshold, score_distribution, sampling, arch, modality)
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
                label=int(labels[i]),
                threshold=threshold,
                predicted=(int(predicted[i]) if predicted is not None else None),
                correct=(bool(correct[i]) if correct is not None else None),
            )


def _write_preamble(
    fh,
    df: pd.DataFrame,
    labels: np.ndarray,
    threshold: Optional[float],
    score_distribution: Optional[Dict[str, Any]],
    sampling: Optional[Dict[str, Any]],
    arch: Optional[str],
    modality: Optional[str],
) -> None:
    fh.write("COSMIC per-variant prediction narrative\n")
    fh.write(_SECTION + "\n")
    fh.write(f"arch            : {arch or '?'}\n")
    fh.write(f"modality        : {modality or '?'}\n")
    fh.write(f"n_total         : {len(df)}\n")
    fh.write(
        f"n_pathogenic    : {int((labels == 1).sum())}\n"
        f"n_neutral       : {int((labels == 0).sum())}\n"
    )
    if sampling:
        fh.write(
            "sampling        : n_per_class={n_per_class} stratify_tier={stratify_tier} seed={seed}\n".format(
                n_per_class=sampling.get("n_per_class"),
                stratify_tier=sampling.get("stratify_tier"),
                seed=sampling.get("seed"),
            )
        )
    if threshold is not None:
        fh.write(f"threshold       : {threshold:+.6f}\n")
    else:
        fh.write("threshold       : (skipped — too few examples per class)\n")
    if score_distribution:
        fh.write("\n-- score distribution --\n")
        for cls in ("neutral", "pathogenic"):
            d = score_distribution.get(cls, {})
            if not d or d.get("n") == 0:
                continue
            fh.write(
                f"  {cls:10s} n={d['n']:>5d}  "
                f"LL(ref)={d['ll_ref_mean']:+.4f} ± {d['ll_ref_std']:.4f}  "
                f"LL(var)={d['ll_var_mean']:+.4f} ± {d['ll_var_std']:.4f}  "
                f"score={d['score_mean']:+.4f} ± {d['score_std']:.4f}\n"
            )

    fh.write("\n")
    fh.write(
        "Scoring rule: score = LL(ref) − LL(var). Higher score means the "
        "model considers the reference sequence more likely than the "
        "variant, which is the expected direction for driver mutations. "
        "The binary prediction marks score > threshold as 'pathogenic' "
        "(driver, FATHMM=DAMAGING / CMC tier 1-2).\n"
    )
    fh.write(_SECTION + "\n\n")


def _pick_preview_indices(
    labels: np.ndarray,
    correct: Optional[np.ndarray],
    total: int,
    preview_count: int,
) -> List[int]:
    if preview_count <= 0:
        return []
    preview_count = min(preview_count, total)

    rng = np.random.default_rng(seed=0)

    if correct is None:
        # No threshold ⇒ class-balanced random preview.
        path_idx = np.where(labels == 1)[0]
        neut_idx = np.where(labels == 0)[0]
        half = preview_count // 2
        pick_path = (
            rng.choice(path_idx, size=min(half, len(path_idx)), replace=False)
            if len(path_idx) else np.array([], dtype=int)
        )
        pick_neut = (
            rng.choice(neut_idx, size=min(preview_count - len(pick_path), len(neut_idx)), replace=False)
            if len(neut_idx) else np.array([], dtype=int)
        )
        picked_arr = np.concatenate([pick_path, pick_neut])
        rng.shuffle(picked_arr)
        return picked_arr.tolist()

    quadrants = {
        ("pathogenic", True): np.where((labels == 1) & (correct == 1))[0],
        ("pathogenic", False): np.where((labels == 1) & (correct == 0))[0],
        ("neutral", True): np.where((labels == 0) & (correct == 1))[0],
        ("neutral", False): np.where((labels == 0) & (correct == 0))[0],
    }

    per_quadrant = max(1, preview_count // 4)
    picked: List[int] = []
    for pool in quadrants.values():
        if len(pool) == 0:
            continue
        take = min(per_quadrant, len(pool))
        drawn = rng.choice(pool, size=take, replace=False)
        picked.extend(int(x) for x in drawn)

    if len(picked) < preview_count:
        remaining = preview_count - len(picked)
        all_indices = np.arange(len(labels))
        leftover = np.setdiff1d(all_indices, np.array(picked))
        if len(leftover):
            drawn = rng.choice(
                leftover, size=min(remaining, len(leftover)), replace=False
            )
            picked.extend(int(x) for x in drawn)

    return sorted(set(picked))


def _write_one_example(
    fh,
    rank: int,
    total_preview: int,
    row: pd.Series,
    row_index: int,
    ll_ref: float,
    ll_var: float,
    score: float,
    label: int,
    threshold: Optional[float],
    predicted: Optional[int],
    correct: Optional[bool],
) -> None:
    label_name = "pathogenic" if label == 1 else "neutral"
    pred_name = (
        "pathogenic" if predicted == 1 else "neutral" if predicted == 0 else "n/a"
    )
    correct_mark = (
        "✓ correct" if correct is True else "✗ wrong" if correct is False else "(no threshold)"
    )

    gene = row.get("GENE_NAME", "?")
    tier = row.get("MUTATION_SIGNIFICANCE_TIER", "?")
    chrom = row.get("chrom", "?")
    pos = row.get("pos", "?")
    ref_allele = row.get("ref", "?")
    alt_allele = row.get("alt", "?")
    fathmm = row.get("FATHMM_PREDICTION", "?")

    ref_seq = str(row.get("reference_sequence", ""))
    var_seq = str(row.get("variant_sequence", ""))

    fh.write(_SECTION + "\n")
    fh.write(
        f"Example {rank} of {total_preview} (dataset index {row_index})\n"
    )
    fh.write(
        f"Gene        : {gene}  (CMC tier {tier})\n"
    )
    fh.write(
        f"Label       : {label_name} (FATHMM={fathmm})\n"
    )
    fh.write(
        f"Prediction  : {pred_name}  {correct_mark}\n"
    )
    fh.write(
        f"Location    : chr{chrom}:{pos}  {ref_allele}>{alt_allele}\n"
    )
    fh.write(_RULE + "\n")
    fh.write(
        f"LL(ref)     : {ll_ref:+.6f}\n"
        f"LL(var)     : {ll_var:+.6f}\n"
        f"score (Δ)   : {score:+.6f}"
    )
    if threshold is not None:
        margin = score - threshold
        fh.write(
            f"   (threshold={threshold:+.6f}, margin={margin:+.6f})\n"
        )
    else:
        fh.write("\n")
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
        caret = "".join("^" if (start + i) in diff_set else " " for i in range(len(ref_chunk)))
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
