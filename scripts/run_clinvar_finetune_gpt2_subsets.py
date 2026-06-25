"""Driver: fine-tune ClinVar classifier on every GPT-2 subset checkpoint.

Companion to run_clinvar_finetune_bert_subsets.py — same shape, same
SKIP_EXISTING caching, same OUT_TAG behaviour, but resolves GPT-2
checkpoint paths and invokes scripts/finetune_clinvar_gpt2.py per run.

Output layout:
    $LEARNING_SOURCE_DIR/genome_sequence/analysis/clinvar_finetune_gpt2/
        gpt2__<subset>__seed<N>/
            metrics.json
            predictions.jsonl
            splits_manifest.csv
            trainer/
        clinvar_finetune_results.csv
        clinvar_finetune_summary.csv
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

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_SUBSETS = [
    "mammal_centered",
    *[f"eukaryote_matched_random_seed{i}" for i in range(1, 11)],
    *[f"global_random_seed{i}" for i in range(1, 11)],
]


def ckpt_path(base_dir: Path, subset: str) -> Path:
    # GPT-2 ckpt root carries both nanoGPT (training_state.bin) and HF
    # (pytorch_model.bin + config.json) formats. The fine-tune script
    # uses the HF format via AutoModelForSequenceClassification, so we
    # pass the ckpt *directory* (not the .bin file).
    return (
        base_dir / "genome_sequence" / "gpt2-output"
        / f"genome_sequence-small-{subset}" / "checkpoint-50000"
    )


def tokenizer_path(base_dir: Path) -> Path:
    return base_dir / "genome_sequence" / "custom_tokenizer_bert_single_nuc"


def run_one(
    subset: str, seed: int, ckpt: Path, tokenizer: Path,
    clinvar_data: Path, out_root: Path,
    num_train_steps: int, learning_rate: float,
    per_device_batch_size: int, max_length: int,
    max_train_rows: int, max_test_rows: int,
    test_chroms: str, device: str,
    python_exe: str, dry_run: bool,
) -> dict:
    out_dir = out_root / f"gpt2__{subset}__seed{seed}"
    cmd = [
        python_exe, "scripts/finetune_clinvar_gpt2.py",
        "--model-path",     str(ckpt),
        "--tokenizer-path", str(tokenizer),
        "--clinvar-data",   str(clinvar_data),
        "--output-dir",     str(out_dir),
        "--num-train-steps", str(num_train_steps),
        "--learning-rate",   str(learning_rate),
        "--per-device-batch-size", str(per_device_batch_size),
        "--max-length",      str(max_length),
        "--max-train-rows",  str(max_train_rows),
        "--max-test-rows",   str(max_test_rows),
        "--test-chroms",     test_chroms,
        "--device",          device,
        "--seed",            str(seed),
    ]
    if dry_run:
        print("DRY:", " ".join(cmd))
        return {"subset": subset, "seed": seed, "auroc": None,
                "run_time_min": 0.0, "status": "DRY_RUN"}

    if not (ckpt / "config.json").exists():
        print(f"SKIP (ckpt missing): {ckpt}", flush=True)
        return {"subset": subset, "seed": seed, "auroc": None,
                "run_time_min": 0.0, "status": "MISSING_CKPT"}

    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open() as f:
                cached = json.load(f)["metrics"]
            auroc = float(cached.get("auroc", float("nan")))
            print(f"SKIP (cached): {subset} seed={seed} AUROC={auroc:.4f}", flush=True)
            return {"subset": subset, "seed": seed, "auroc": auroc,
                    "run_time_min": 0.0, "status": "CACHED"}
        except (KeyError, json.JSONDecodeError, ValueError):
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    t0 = time.time()
    print(f"RUN: gpt2 {subset} seed={seed} ...", flush=True)
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT),
        )
    elapsed_min = (time.time() - t0) / 60.0

    if proc.returncode != 0 or not metrics_path.exists():
        print(f"  ✗ FAILED ({elapsed_min:.1f} min). See {log_path}", flush=True)
        return {"subset": subset, "seed": seed, "auroc": None,
                "run_time_min": elapsed_min, "status": "FAILED"}

    with metrics_path.open() as f:
        m = json.load(f)["metrics"]
    auroc = float(m.get("auroc", float("nan")))
    print(f"  ✓ AUROC={auroc:.4f} ({elapsed_min:.1f} min)", flush=True)
    return {"subset": subset, "seed": seed, "auroc": auroc,
            "run_time_min": elapsed_min, "status": "OK"}


def write_summary(results: list[dict], out_path: Path) -> None:
    g = defaultdict(list)
    for r in results:
        if r["status"] in ("OK", "CACHED") and r["auroc"] is not None:
            g[r["subset"]].append(r["auroc"])
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "subset", "model_type", "n_seeds",
            "auroc_mean", "auroc_std", "auroc_min", "auroc_max",
        ])
        w.writeheader()
        for s, vals in sorted(g.items()):
            if not vals:
                continue
            w.writerow({
                "subset": s, "model_type": "gpt2", "n_seeds": len(vals),
                "auroc_mean": round(mean(vals), 4),
                "auroc_std": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
                "auroc_min": round(min(vals), 4),
                "auroc_max": round(max(vals), 4),
            })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=os.environ.get("LEARNING_SOURCE_DIR"))
    ap.add_argument("--clinvar-data", default=None)
    ap.add_argument("--subsets", default=None)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--num-train-steps", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--per-device-batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--max-train-rows", type=int, default=20_000)
    ap.add_argument("--max-test-rows", type=int, default=4_000)
    ap.add_argument("--test-chroms", default="8,X,Y")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-tag", default=None)
    ap.add_argument(
        "--python-exe",
        default=os.environ.get(
            "PYTHON", str(Path.home() / "miniforge3/envs/molcrawl/bin/python")
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.base_dir:
        ap.error("--base-dir required (or set LEARNING_SOURCE_DIR)")
    base_dir = Path(args.base_dir)
    clinvar_data = Path(args.clinvar_data) if args.clinvar_data else (
        base_dir / "genome_sequence" / "clinvar" / "clinvar_sequences.csv"
    )
    out_dir_name = (
        f"clinvar_finetune_gpt2_{args.out_tag}" if args.out_tag
        else "clinvar_finetune_gpt2"
    )
    out_root = base_dir / "genome_sequence" / "analysis" / out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    subsets = (
        [s for s in args.subsets.split(",") if s] if args.subsets
        else [s for s in ALL_SUBSETS if (ckpt_path(base_dir, s) / "config.json").exists()]
    )
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tokenizer = tokenizer_path(base_dir)
    if not tokenizer.exists():
        print(f"ERROR: tokenizer dir missing: {tokenizer}", file=sys.stderr)
        return 1

    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  base_dir     = {base_dir}")
    print(f"  out_root     = {out_root}")
    print(f"  clinvar_data = {clinvar_data}")
    print(f"  subsets      = {len(subsets)}")
    print(f"  seeds        = {seeds}")
    print(f"  steps        = {args.num_train_steps}")
    print(f"  device       = {args.device}")
    print(f"  total runs   = {len(subsets) * len(seeds)}")
    print()

    results: list[dict] = []
    t_total = time.time()
    for s in subsets:
        ckpt = ckpt_path(base_dir, s)
        for seed in seeds:
            r = run_one(
                subset=s, seed=seed, ckpt=ckpt, tokenizer=tokenizer,
                clinvar_data=clinvar_data, out_root=out_root,
                num_train_steps=args.num_train_steps,
                learning_rate=args.learning_rate,
                per_device_batch_size=args.per_device_batch_size,
                max_length=args.max_length,
                max_train_rows=args.max_train_rows,
                max_test_rows=args.max_test_rows,
                test_chroms=args.test_chroms,
                device=args.device,
                python_exe=args.python_exe, dry_run=args.dry_run,
            )
            results.append(r)
            if not args.dry_run:
                with (out_root / "clinvar_finetune_results.csv").open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=[
                        "subset", "seed", "auroc", "run_time_min", "status",
                    ])
                    w.writeheader()
                    for r2 in results:
                        w.writerow(r2)

    write_summary(results, out_root / "clinvar_finetune_summary.csv")

    total_min = (time.time() - t_total) / 60.0
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_cached = sum(1 for r in results if r["status"] == "CACHED")
    n_fail = sum(1 for r in results if r["status"] not in ("OK", "CACHED", "DRY_RUN"))
    print()
    print(f"[done] total {total_min:.1f} min")
    print(f"  ok       : {n_ok}")
    print(f"  cached   : {n_cached}")
    print(f"  failed   : {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
