"""Run ClinVar zero-shot evaluation across all subset checkpoints.

For each (subset, model_type, seed) tuple, calls
``python -m molcrawl.tasks.evaluation.clinvar`` with the appropriate
``--model-path`` / ``--tokenizer-path`` / ``--clinvar-data`` arguments,
then aggregates per-run metrics.json files into two CSVs and a text
report following the format specified in
``tmp/docs_tmp_local/yigarashi-issue/clinvar-auroc-evaluation-instructions.md``.

The evaluation uses pseudo-log-likelihood scoring (zero-shot) — no
fine-tuning. This is faster than the fine-tune approach the instruction
document outlined as the primary path, and is methodologically cleaner
for comparing how pretraining-corpus differences propagate to downstream
discrimination (no fine-tuning noise to confound the comparison).

Seeds: each (subset, model) combination is run with multiple seeds
(default {1, 2, 3}). Because evaluation is zero-shot, the seed only
controls the per-class chromosome-stratified sampling — so multiple
seeds give independent AUROC estimates that can be averaged and
std'd, matching the instruction document's "3-seed AUROC mean ± std"
convention.

Usage:
    python scripts/run_clinvar_eval_subsets.py [--dry-run]
                                                [--subsets ...]
                                                [--models bert,gpt2]
                                                [--seeds 1,2,3]
                                                [--n-per-class N]
                                                [--device cuda|cpu]

The output dir is fixed to:
    <LEARNING_SOURCE_DIR>/genome_sequence/analysis/clinvar_evaluation/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_SUBSETS_FIRST_BATCH = [
    "mammal_centered",
    "eukaryote_matched_random_seed1",
    "eukaryote_matched_random_seed2",
    "eukaryote_matched_random_seed3",
    "eukaryote_matched_random_seed4",
    "eukaryote_matched_random_seed5",
    "global_random_seed1",
    "global_random_seed2",
    "global_random_seed3",
]
SECOND_BATCH_SUBSETS = [
    "eukaryote_matched_random_seed6",
    "eukaryote_matched_random_seed7",
    "eukaryote_matched_random_seed8",
    "eukaryote_matched_random_seed9",
    "eukaryote_matched_random_seed10",
    "global_random_seed4",
    "global_random_seed5",
    "global_random_seed6",
    "global_random_seed7",
    "global_random_seed8",
    "global_random_seed9",
    "global_random_seed10",
]
ALL_SUBSETS = ALL_SUBSETS_FIRST_BATCH + SECOND_BATCH_SUBSETS


def ckpt_path(base_dir: Path, subset: str, model_type: str) -> Path:
    """Return the checkpoint file/dir path for a given (subset, model)."""
    if model_type == "bert":
        return (
            base_dir / "genome_sequence" / "bert-output"
            / f"genome_sequence-small-{subset}" / "checkpoint-60000"
        )
    elif model_type == "gpt2":
        return (
            base_dir / "genome_sequence" / "gpt2-output"
            / f"genome_sequence-small-{subset}" / "checkpoint-50000"
            / "training_state.bin"
        )
    else:
        raise ValueError(f"unknown model_type: {model_type}")


def tokenizer_path(base_dir: Path) -> Path:
    """Subset BERT/GPT-2 use the same single-nucleotide HF tokenizer."""
    return base_dir / "genome_sequence" / "custom_tokenizer_bert_single_nuc"


def run_one(
    subset: str,
    model_type: str,
    seed: int,
    ckpt: Path,
    tokenizer: Path,
    clinvar_data: Path,
    out_root: Path,
    n_per_class: int,
    device: str,
    context_length: int,
    python_exe: str,
    dry_run: bool,
    score_window_half: Optional[int] = None,
    flank: int = 64,
) -> dict:
    """Run a single evaluation; return a dict for clinvar_auroc_results.csv."""
    out_dir = out_root / f"{model_type}__{subset}__seed{seed}"
    cmd = [
        python_exe, "-m", "molcrawl.tasks.evaluation.clinvar",
        "--model-path", str(ckpt),
        "--tokenizer-path", str(tokenizer),
        "--clinvar-data", str(clinvar_data),
        "--output-dir", str(out_dir),
        "--arch", model_type,
        "--modality", "genome_sequence",
        "--n-per-class", str(n_per_class),
        "--device", device,
        "--seed", str(seed),
        "--context-length", str(context_length),
    ]
    if score_window_half is not None:
        cmd += ["--score-window-half", str(score_window_half),
                "--flank", str(flank)]
    if dry_run:
        print("DRY:", " ".join(cmd))
        return {
            "subset": subset,
            "model_type": model_type,
            "seed": seed,
            "auroc": None,
            "run_time_min": 0.0,
            "status": "DRY_RUN",
        }

    if not ckpt.exists():
        print(f"SKIP (ckpt missing): {ckpt}", flush=True)
        return {
            "subset": subset, "model_type": model_type, "seed": seed,
            "auroc": None, "run_time_min": 0.0, "status": "MISSING_CKPT",
        }

    # Skip if a completed run already exists. Lets a re-invocation that
    # adds new seeds (e.g. --seeds 1,2,3 after the first seed=1-only
    # SLURM run) reuse the seed=1 outputs instead of recomputing them.
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open() as f:
                existing = json.load(f)["metrics"]
            cached_auroc = float(existing.get("auroc", float("nan")))
            print(
                f"SKIP (cached): {model_type} {subset} seed={seed} "
                f"AUROC={cached_auroc:.4f}",
                flush=True,
            )
            return {
                "subset": subset, "model_type": model_type, "seed": seed,
                "auroc": cached_auroc, "run_time_min": 0.0, "status": "CACHED",
            }
        except (KeyError, json.JSONDecodeError, ValueError):
            # Malformed cached metrics → fall through to a fresh run.
            print(
                f"WARN: metrics.json at {metrics_path} is unreadable; "
                "re-running.",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    t0 = time.time()
    print(f"RUN: {model_type} {subset} seed={seed} ...", flush=True)
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    elapsed_min = (time.time() - t0) / 60.0

    metrics_path = out_dir / "metrics.json"
    if proc.returncode != 0 or not metrics_path.exists():
        print(f"  ✗ FAILED ({elapsed_min:.1f} min, exit={proc.returncode}). "
              f"See {log_path}", flush=True)
        return {
            "subset": subset, "model_type": model_type, "seed": seed,
            "auroc": None, "run_time_min": elapsed_min, "status": "FAILED",
        }

    with metrics_path.open() as f:
        metrics = json.load(f)["metrics"]
    auroc = float(metrics.get("auroc", float("nan")))
    print(f"  ✓ AUROC={auroc:.4f} ({elapsed_min:.1f} min)", flush=True)
    return {
        "subset": subset, "model_type": model_type, "seed": seed,
        "auroc": auroc, "run_time_min": elapsed_min, "status": "OK",
    }


def write_results_csv(results: list[dict], out_path: Path) -> None:
    fields = ["subset", "model_type", "seed", "auroc", "run_time_min", "status"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)


# Status values whose AUROC should flow into summary.csv / report.txt.
# CACHED rows replay a prior successful run's metrics.json — they carry a
# valid auroc and must aggregate the same as a fresh OK row, otherwise a
# re-invocation that adds new seeds wipes the earlier seeds out of the
# summary even though their per-run metrics.json is still on disk.
_AGGREGATABLE_STATUS = ("OK", "CACHED")


def write_summary_csv(results: list[dict], out_path: Path) -> None:
    """Aggregate over seeds → (subset, model) AUROC mean/std/min/max."""
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        if r["status"] not in _AGGREGATABLE_STATUS or r["auroc"] is None:
            continue
        by_key[(r["subset"], r["model_type"])].append(r["auroc"])
    fields = ["subset", "model_type", "n_seeds",
              "auroc_mean", "auroc_std", "auroc_min", "auroc_max"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (subset, model_type), vals in sorted(by_key.items()):
            if len(vals) == 0:
                continue
            w.writerow({
                "subset": subset, "model_type": model_type,
                "n_seeds": len(vals),
                "auroc_mean": round(mean(vals), 4),
                "auroc_std": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
                "auroc_min": round(min(vals), 4),
                "auroc_max": round(max(vals), 4),
            })


def write_report_txt(results: list[dict], summary_path: Path, out_path: Path) -> None:
    """Human-readable text report matching the instruction template."""
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    subsets_seen: set[str] = set()
    for r in results:
        if r["status"] in _AGGREGATABLE_STATUS and r["auroc"] is not None:
            by_key[(r["subset"], r["model_type"])].append(r["auroc"])
            subsets_seen.add(r["subset"])

    lines: list[str] = []
    lines.append("ClinVar AUROC — subset pretrain checkpoints (zero-shot PLL)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Generated at:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Results CSV:   {summary_path}")
    lines.append("")
    lines.append(f"{'subset':40s} {'BERT (mean±std)':>18s} {'GPT-2 (mean±std)':>18s}")
    lines.append("-" * 78)
    # Preserve canonical subset order
    for subset in ALL_SUBSETS:
        if subset not in subsets_seen:
            continue
        bert_vals = by_key.get((subset, "bert"), [])
        gpt2_vals = by_key.get((subset, "gpt2"), [])
        bert_cell = (
            f"{mean(bert_vals):.4f}±{stdev(bert_vals):.4f}"
            if len(bert_vals) > 1
            else (f"{bert_vals[0]:.4f}" if bert_vals else "-")
        )
        gpt2_cell = (
            f"{mean(gpt2_vals):.4f}±{stdev(gpt2_vals):.4f}"
            if len(gpt2_vals) > 1
            else (f"{gpt2_vals[0]:.4f}" if gpt2_vals else "-")
        )
        lines.append(f"{subset:40s} {bert_cell:>18s} {gpt2_cell:>18s}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - AUROC computed by zero-shot pseudo-log-likelihood (no fine-tuning).")
    lines.append("  - Per-class chromosome-stratified sampling controls the per-chrom")
    lines.append("    pathogenic-rate variance (overall ~27%, chrX ~48%, chrY ~86%).")
    lines.append("  - Multiple seeds vary the sampled variants only (model fixed).")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default=os.environ.get("LEARNING_SOURCE_DIR"),
        help=(
            "Root holding genome_sequence/{bert,gpt2}-output and "
            "genome_sequence/analysis/. Defaults to $LEARNING_SOURCE_DIR; "
            "required when that env var is unset."
        ),
    )
    ap.add_argument(
        "--clinvar-data",
        default=None,
        help=(
            "ClinVar source CSV (reference_sequence, variant_sequence, ...). "
            "Defaults to $LEARNING_SOURCE_DIR/genome_sequence/clinvar/"
            "clinvar_sequences.csv if not given."
        ),
    )
    ap.add_argument(
        "--subsets", default=None,
        help="Comma-separated subset names. Default: all currently available "
             "(checked via ckpt existence).",
    )
    ap.add_argument("--models", default="bert,gpt2", help="Comma-separated: bert,gpt2")
    ap.add_argument("--seeds", default="1,2,3",
                    help="Comma-separated seeds (default: 1,2,3 → mean±std)")
    ap.add_argument("--n-per-class", type=int, default=1000,
                    help="ClinVar sample size per class (default: 1000)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument(
        "--score-window-half",
        type=int,
        default=None,
        help=(
            "Forward to clinvar evaluator: restrict PLL averaging to "
            "±N tokens around the variant centre (default = full "
            "sequence). 32 is a sensible value for the 128-nt window "
            "produced by download_clinvar_sequences."
        ),
    )
    ap.add_argument(
        "--flank",
        type=int,
        default=64,
        help="Variant centre position (default 64). Only used when --score-window-half is set.",
    )
    ap.add_argument(
        "--out-tag",
        default=None,
        help=(
            "Optional suffix for the analysis output dir, used to keep "
            "parallel sweeps (e.g. full-vs-window, full-vs-2star) side "
            "by side instead of overwriting each other. Default = "
            "'clinvar_evaluation' (unchanged historical layout)."
        ),
    )
    ap.add_argument(
        "--python-exe",
        default=os.environ.get(
            "PYTHON", str(Path.home() / "miniforge3/envs/molcrawl/bin/python")
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.base_dir:
        ap.error(
            "--base-dir is required (or set LEARNING_SOURCE_DIR in env)."
        )
    base_dir = Path(args.base_dir)
    if args.clinvar_data:
        clinvar_data = Path(args.clinvar_data)
    else:
        clinvar_data = (
            base_dir / "genome_sequence" / "clinvar" / "clinvar_sequences.csv"
        )
    out_dir_name = (
        f"clinvar_evaluation_{args.out_tag}" if args.out_tag
        else "clinvar_evaluation"
    )
    out_root = base_dir / "genome_sequence" / "analysis" / out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    if args.subsets:
        subsets = [s for s in args.subsets.split(",") if s]
    else:
        # Auto-detect from existing ckpts (BERT dir is the reference).
        subsets = []
        for s in ALL_SUBSETS:
            bert_ckpt = ckpt_path(base_dir, s, "bert")
            gpt2_ckpt = ckpt_path(base_dir, s, "gpt2")
            if bert_ckpt.exists() or gpt2_ckpt.exists():
                subsets.append(s)
    models = [m for m in args.models.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    tokenizer = tokenizer_path(base_dir)
    if not tokenizer.exists():
        print(f"ERROR: tokenizer not found at {tokenizer}", file=sys.stderr)
        return 1

    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  base_dir      = {base_dir}")
    print(f"  out_root      = {out_root}")
    print(f"  subsets ({len(subsets)}) = {subsets[:3]}...")
    print(f"  models        = {models}")
    print(f"  seeds         = {seeds}")
    print(f"  n_per_class   = {args.n_per_class}")
    print(f"  device        = {args.device}")
    print(f"  total runs    = {len(subsets) * len(models) * len(seeds)}")
    print()

    results: list[dict] = []
    t_total = time.time()

    for subset in subsets:
        for model_type in models:
            ckpt = ckpt_path(base_dir, subset, model_type)
            for seed in seeds:
                r = run_one(
                    subset=subset, model_type=model_type, seed=seed,
                    ckpt=ckpt, tokenizer=tokenizer,
                    clinvar_data=clinvar_data, out_root=out_root,
                    n_per_class=args.n_per_class, device=args.device,
                    context_length=args.context_length,
                    python_exe=args.python_exe, dry_run=args.dry_run,
                    score_window_half=args.score_window_half,
                    flank=args.flank,
                )
                results.append(r)
                # Incrementally rewrite the CSV so partial progress isn't lost
                if not args.dry_run:
                    write_results_csv(results, out_root / "clinvar_auroc_results.csv")

    if args.dry_run:
        print()
        print(f"[dry-run] would run {len(results)} evaluations")
        return 0

    write_results_csv(results, out_root / "clinvar_auroc_results.csv")
    write_summary_csv(results, out_root / "clinvar_auroc_summary.csv")
    write_report_txt(results, out_root / "clinvar_auroc_summary.csv",
                     out_root / "clinvar_auroc_report.txt")

    total_min = (time.time() - t_total) / 60.0
    n_ok     = sum(1 for r in results if r["status"] == "OK")
    n_cached = sum(1 for r in results if r["status"] == "CACHED")
    n_fail   = sum(1 for r in results
                   if r["status"] not in ("OK", "CACHED", "DRY_RUN"))
    print()
    print(f"[done] total {total_min:.1f} min")
    print(f"  ok       : {n_ok}")
    print(f"  cached   : {n_cached}")
    print(f"  failed   : {n_fail}")
    print(f"  results  : {out_root / 'clinvar_auroc_results.csv'}")
    print(f"  summary  : {out_root / 'clinvar_auroc_summary.csv'}")
    print(f"  report   : {out_root / 'clinvar_auroc_report.txt'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
