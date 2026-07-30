"""Filter ClinVar CSV by review_status, joining against the gonzalobenegas/clinvar
HuggingFace cache that ships the upstream review_status column.

The current ``clinvar_sequences.csv`` does not carry ``review_status`` (the
upcoming PR ``feat/clinvar-csv-vcv-metadata`` adds it). Until that PR lands
and the CSV is regenerated, this script reconstructs the review_status
side-channel by joining ``(chrom, pos, ref, alt)`` against the local HF
cache, then emits a filtered CSV restricted to the requested
gold-star tier(s).

Usage:
    python scripts/filter_clinvar_by_review_status.py \
        --in-csv  /path/to/clinvar_sequences.csv \
        --out-csv /path/to/clinvar_sequences_high_quality.csv \
        --min-stars 2

Star mapping (matches ClinVar review_status enum):
    0 = no assertion criteria provided
    1 = criteria provided, single submitter
        (and criteria provided, conflicting interpretations)
    2 = criteria provided, multiple submitters, no conflicts
    3 = reviewed by expert panel
    4 = practice guideline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset


REVIEW_STATUS_STAR = {
    "no_assertion_provided":                                          0,
    "no_assertion_criteria_provided":                                 0,
    "no_classification_provided":                                     0,
    "no_interpretation_for_the_single_variant":                       0,
    "criteria_provided,_conflicting_interpretations":                 1,
    "criteria_provided,_conflicting_classifications":                 1,
    "criteria_provided,_single_submitter":                            1,
    "criteria_provided,_multiple_submitters,_no_conflicts":           2,
    "reviewed_by_expert_panel":                                       3,
    "practice_guideline":                                             4,
}


def star_of(rs: str) -> int:
    """Return the gold-star tier for a review_status string."""
    if rs is None or (isinstance(rs, float) and pd.isna(rs)):
        return 0
    return REVIEW_STATUS_STAR.get(str(rs), 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-csv", required=True, help="Source ClinVar CSV.")
    ap.add_argument("--out-csv", required=True, help="Filtered CSV destination.")
    ap.add_argument(
        "--min-stars", type=int, default=2,
        help="Minimum gold-star tier to keep (default 2).",
    )
    ap.add_argument(
        "--hf-dataset", default="gonzalobenegas/clinvar",
        help="HF dataset to join against for review_status.",
    )
    args = ap.parse_args()

    src = Path(args.in_csv)
    dst = Path(args.out_csv)
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1

    print(f"[1/4] loading source CSV: {src}", flush=True)
    csv = pd.read_csv(src)
    print(f"      rows: {len(csv):,}", flush=True)

    print(f"[2/4] loading HF dataset: {args.hf_dataset}", flush=True)
    hf = load_dataset(args.hf_dataset)["test"].to_pandas()
    print(f"      rows: {len(hf):,}", flush=True)

    print("[3/4] joining on (chrom, pos, ref, alt) …", flush=True)
    # Normalise types for the join key.
    for df in (csv, hf):
        df["chrom"] = df["chrom"].astype(str)
        df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
        df["ref"] = df["ref"].astype(str)
        df["alt"] = df["alt"].astype(str)

    merged = csv.merge(
        hf[["chrom", "pos", "ref", "alt", "id", "review_status", "consequence"]],
        on=["chrom", "pos", "ref", "alt"],
        how="left",
    )
    # Pre-format vcv_id while we're here (downstream evaluators are agnostic
    # to the new columns thanks to the existing whitelist contract).
    merged["vcv_id"] = merged["id"].apply(
        lambda v: f"VCV{int(v):09d}" if pd.notna(v) else None
    )
    merged = merged.drop(columns=["id"])
    merged["star"] = merged["review_status"].apply(star_of)

    matched = merged["review_status"].notna().sum()
    print(
        f"      matched {matched:,} / {len(merged):,} rows "
        f"({100*matched/max(len(merged),1):.1f}%)",
        flush=True,
    )

    print(f"[4/4] filtering star >= {args.min_stars} …", flush=True)
    filtered = merged[merged["star"] >= args.min_stars].copy()
    print(
        f"      kept {len(filtered):,} rows "
        f"({100*len(filtered)/max(len(merged),1):.1f}% of joined data)",
        flush=True,
    )

    # Star tier distribution (informational).
    print("\nstar tier distribution of input (after join):")
    tier_dist = merged["star"].value_counts().sort_index()
    for star, n in tier_dist.items():
        print(f"  star={star} : {n:>7,} rows")
    print("\nstar tier distribution of output (filtered):")
    tier_dist = filtered["star"].value_counts().sort_index()
    for star, n in tier_dist.items():
        print(f"  star={star} : {n:>7,} rows")

    # Drop the working ``star`` column before writing (downstream readers
    # only know review_status / vcv_id / consequence).
    filtered = filtered.drop(columns=["star"])
    # Put vcv_id / metadata at the front (matches the upcoming
    # feat/clinvar-csv-vcv-metadata column order).
    front_cols = ["vcv_id", "review_status", "consequence"]
    other_cols = [c for c in filtered.columns if c not in front_cols]
    filtered = filtered[front_cols + other_cols]

    dst.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(dst, index=False)
    print(f"\nwrote {len(filtered):,} rows to {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
