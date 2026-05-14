"""Per-generated-sequence dumps for the protein foldability evaluator.

JSONL: one record per generated sequence (raw + cleaned + length +
per-AA composition + novelty flag). predictions.txt previews three
quadrants — novel-long / novel-short / duplicate-of-reference — so
reviewers can see actual generated sequences instead of relying on
aggregated metrics alone.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from .metrics import AA_ALPHABET

logger = logging.getLogger(__name__)


def write_predictions(
    output_dir: Path,
    raw_generated: Sequence[str],
    cleaned_generated: Sequence[str],
    reference_set: Set[str],
    sampling_params: Optional[Dict[str, Any]] = None,
    arch: Optional[str] = None,
    preview_count: int = 30,
    foldable_min_length: int = 50,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "predictions.jsonl"
    _write_jsonl(jsonl_path, raw_generated, cleaned_generated, reference_set)

    txt_path = output_dir / "predictions.txt"
    _write_narrative(
        txt_path,
        raw_generated=raw_generated,
        cleaned_generated=cleaned_generated,
        reference_set=reference_set,
        sampling_params=sampling_params,
        arch=arch,
        preview_count=preview_count,
        foldable_min_length=foldable_min_length,
    )

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


# ---------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------


def _aa_counts(seq: str) -> Dict[str, int]:
    counter: Counter = Counter()
    for ch in seq:
        ch = ch.upper()
        if ch in AA_ALPHABET:
            counter[ch] += 1
    return {aa: int(counter.get(aa, 0)) for aa in AA_ALPHABET}


def _write_jsonl(
    path: Path,
    raw_generated: Sequence[str],
    cleaned_generated: Sequence[str],
    reference_set: Set[str],
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, (raw, cleaned) in enumerate(zip(raw_generated, cleaned_generated)):
            in_ref = cleaned in reference_set
            record: Dict[str, Any] = {
                "index": int(i),
                "generated_raw": str(raw),
                "cleaned": str(cleaned),
                "length": int(len(cleaned)),
                "novel_vs_reference": (not in_ref) if cleaned else None,
                "aa_counts": _aa_counts(cleaned),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    raw_generated: Sequence[str],
    cleaned_generated: Sequence[str],
    reference_set: Set[str],
    sampling_params: Optional[Dict[str, Any]],
    arch: Optional[str],
    preview_count: int,
    foldable_min_length: int,
) -> None:
    n = len(raw_generated)
    novel_idx = [
        i for i, c in enumerate(cleaned_generated) if c and c not in reference_set
    ]
    duplicate_idx = [
        i for i, c in enumerate(cleaned_generated) if c and c in reference_set
    ]

    novel_long = [i for i in novel_idx if len(cleaned_generated[i]) >= foldable_min_length]
    novel_short = [i for i in novel_idx if len(cleaned_generated[i]) < foldable_min_length]

    lengths = np.array([len(c) for c in cleaned_generated], dtype=float)

    with path.open("w", encoding="utf-8") as fh:
        fh.write("Protein foldability per-sequence narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch              : {arch or '?'}\n")
        fh.write(f"n_generated       : {n}\n")
        fh.write(
            f"n_novel           : {len(novel_idx)}  "
            f"(={len(novel_idx) / max(n, 1):.3f} of total)\n"
        )
        fh.write(
            f"n_novel_long      : {len(novel_long)}  "
            f"(>= {foldable_min_length} aa, foldable-likely)\n"
        )
        fh.write(
            f"n_novel_short     : {len(novel_short)}  "
            f"(< {foldable_min_length} aa, peptide-only)\n"
        )
        fh.write(
            f"n_duplicate_of_ref: {len(duplicate_idx)}\n"
        )
        if lengths.size:
            fh.write(
                f"length stats      : mean={lengths.mean():.1f} std={lengths.std():.1f} "
                f"min={lengths.min():.0f} median={np.median(lengths):.0f} "
                f"max={lengths.max():.0f}\n"
            )
        if sampling_params:
            fh.write(f"sampling_params   : {sampling_params}\n")

        fh.write("\n")
        fh.write(
            "Score blocks below sample three quadrants of generation behaviour: "
            "(A) novel + long  = candidate de-novo proteins above the foldable "
            "size threshold; (B) novel + short = peptide-scale outputs that "
            "rarely fold by themselves; (C) duplicate = exact copies of the "
            "reference corpus.\n"
        )
        fh.write(_SECTION + "\n\n")

        per_quadrant = max(1, preview_count // 3)
        rng = np.random.default_rng(seed=0)

        def _draw(pool: Sequence[int], take: int) -> List[int]:
            if not pool:
                return []
            take = min(take, len(pool))
            return [int(x) for x in rng.choice(pool, size=take, replace=False)]

        rank = 0
        for label, pool in [
            ("novel + long", novel_long),
            ("novel + short", novel_short),
            ("duplicate of reference", duplicate_idx),
        ]:
            for i in _draw(pool, per_quadrant):
                rank += 1
                _write_one_example(
                    fh,
                    rank=rank,
                    label=label,
                    raw=raw_generated[i],
                    cleaned=cleaned_generated[i],
                )


def _write_one_example(
    fh,
    rank: int,
    label: str,
    raw: str,
    cleaned: str,
) -> None:
    counts = _aa_counts(cleaned)
    top_aa = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    top_aa_str = " ".join(f"{aa}={c}" for aa, c in top_aa if c > 0)
    fh.write(_SECTION + "\n")
    fh.write(f"Example {rank}  ({label})\n")
    fh.write(_RULE + "\n")
    fh.write(f"length    : {len(cleaned)}\n")
    fh.write(f"top AAs   : {top_aa_str}\n")
    fh.write(f"raw       : {raw[:200]}{'...' if len(raw) > 200 else ''}\n")
    # Wrap the cleaned sequence at 60 chars/line for readability
    fh.write("cleaned   :\n")
    for offset in range(0, len(cleaned), 60):
        fh.write(f"  [{offset:>4d}] {cleaned[offset:offset + 60]}\n")
    fh.write(_SECTION + "\n\n")
