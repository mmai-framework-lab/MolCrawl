"""Re-run Phase 4 (parquet → Arrow) for all 21 Evo2 subsets with the new
shuffled split, without re-touching Phases 1-3.

Background:
  The original ``process4_subset_parquet_to_arrow`` (preparation.py) split
  the concatenated dataset by accession-ordered offsets, which produced
  identical valid/test sets across subsets that share a large tail parquet
  (e.g. eukaryote_seed1/3/4 all included GCF_949987535.1, the alphabetically
  last and largest file). See
  tmp/docs_tmp_local/yigarashi-issue/20260615-seed134-investigation-followup.md.

  The fix adds ``ds = ds.shuffle(seed=42)`` before the split. This script
  re-runs that single phase for every subset, replacing the old
  ``training_ready_hf_dataset_{bert,gpt2}/`` directories in place.

⚠️ SAFETY:
  This script overwrites Arrow files that may be memory-mapped by running
  SLURM training jobs → those jobs will crash with SIGBUS or similar.
  DO NOT run while any training job is actively reading from
  ``training_ready_hf_dataset_*``. Check ``squeue -u $USER`` and either wait
  for completion or ``scancel`` first.

Usage:
  python scripts/rerun_phase4_subset_arrow.py \
      --base-dir $LEARNING_SOURCE_DIR/genome_sequence \
      [--subsets mammal_centered,eukaryote_matched_random_seed1,...] \
      [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from molcrawl.data.genome_sequence.preparation import (
    process4_subset_parquet_to_arrow,
)

# Make repo importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ALL_SUBSETS = [
    "mammal_centered",
    "eukaryote_matched_random_seed1",
    "eukaryote_matched_random_seed2",
    "eukaryote_matched_random_seed3",
    "eukaryote_matched_random_seed4",
    "eukaryote_matched_random_seed5",
    "eukaryote_matched_random_seed6",
    "eukaryote_matched_random_seed7",
    "eukaryote_matched_random_seed8",
    "eukaryote_matched_random_seed9",
    "eukaryote_matched_random_seed10",
    "global_random_seed1",
    "global_random_seed2",
    "global_random_seed3",
    "global_random_seed4",
    "global_random_seed5",
    "global_random_seed6",
    "global_random_seed7",
    "global_random_seed8",
    "global_random_seed9",
    "global_random_seed10",
]
MODELS = ["bert", "gpt2"]


def cleanup_previous(subset_dir: Path) -> None:
    """Remove the old Arrow output dirs and the completion marker so that
    process4 sees a clean slate and writes fresh shuffled splits.

    save_to_disk on an existing populated dir would either error out or
    interleave files; explicit cleanup is the simplest safe path.
    """
    for model in MODELS:
        arrow_dir = subset_dir / f"training_ready_hf_dataset_{model}"
        if arrow_dir.exists():
            shutil.rmtree(arrow_dir)
            print(f"    removed: {arrow_dir}")
    marker = subset_dir / "parquet_to_arrow_complete.marker"
    if marker.exists():
        marker.unlink()
        print(f"    removed marker: {marker}")


def _default_base_dir() -> str | None:
    """`$LEARNING_SOURCE_DIR/genome_sequence` if the env var is set."""
    env = os.environ.get("LEARNING_SOURCE_DIR")
    return str(Path(env) / "genome_sequence") if env else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-dir",
        default=_default_base_dir(),
        help=(
            "Genome-sequence preparation root (the dir holding per-subset "
            "subdirs). Defaults to $LEARNING_SOURCE_DIR/genome_sequence; "
            "required when that env var is unset."
        ),
    )
    ap.add_argument(
        "--subsets",
        default=",".join(ALL_SUBSETS),
        help="Comma-separated subset names (default: all 21)",
    )
    ap.add_argument("--dry-run", action="store_true", help="List what would be done")
    args = ap.parse_args()

    if not args.base_dir:
        ap.error(
            "--base-dir is required (or set LEARNING_SOURCE_DIR in env)."
        )
    base = Path(args.base_dir)
    subsets = [s for s in args.subsets.split(",") if s]

    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  base_dir = {base}")
    print(f"  subsets  = {len(subsets)} ({', '.join(subsets[:3])}...)")
    print(f"  dry_run  = {args.dry_run}")
    print()

    if args.dry_run:
        for s in subsets:
            subset_dir = base / s
            arrow_paths = [
                subset_dir / f"training_ready_hf_dataset_{m}" for m in MODELS
            ]
            print(f"DRY: {s}")
            for p in arrow_paths:
                exists = "exists" if p.exists() else "missing"
                print(f"    would rewrite: {p} ({exists})")
        return 0

    failures: list[str] = []
    t_total = time.time()

    for i, subset in enumerate(subsets, 1):
        subset_dir = base / subset
        if not subset_dir.exists():
            print(f"[{i:2d}/{len(subsets)}] SKIP (no subset dir): {subset}", flush=True)
            failures.append(subset)
            continue

        print(f"[{i:2d}/{len(subsets)}] {subset}", flush=True)
        t0 = time.time()

        # 1) clean previous Arrow + marker
        cleanup_previous(subset_dir)

        # 2) re-run Phase 4
        ok = process4_subset_parquet_to_arrow(
            base_dir=str(subset_dir),
            models=MODELS,
            valid_size=50_000,
            test_size=50_000,
            valid_frac=0.005,
            test_frac=0.005,
            remove_parquet=False,   # keep parquet for safety / re-runs
            force=False,            # marker was removed above, no need to force
        )

        elapsed = time.time() - t0
        status = "OK" if ok else "FAILED"
        print(f"    → {status} ({elapsed:.0f} s)", flush=True)
        if not ok:
            failures.append(subset)

    print()
    print(f"[done] total {time.time() - t_total:.0f} s")
    if failures:
        print(f"  failures: {failures}")
        return 1
    print(f"  all {len(subsets)} subsets re-written successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
