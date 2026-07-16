"""42 subset config を全部 eval して resolved 値を確認 (readiness 用オフライン検証).

For each of 21 subsets × 2 archs = 42 configs, evals the config module with
GENOME_SUBSET set, prints the resolved (global_batch, LR, max_iters/steps,
warmup, early_stopping, dataset_dir) so the readiness report can carry
the full table.

Usage:
    python3 tmp/scripts/autopilot/analyzers/verify_subset_configs_resolved.py \
        --subsets-file tmp/scripts/autopilot/state/subsets_21.txt \
        --learning-source /lustre/home/matsubara/learning_source_20260710_genome_v2 \
        --out-md tmp/scripts/autopilot/state/subset_configs_resolved.md
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

ROOT = Path("/lustre/home/matsubara/riken-dataset-fundational-model")


def evalcfg(cfg_path: Path, arch: str, subset: str) -> dict:
    os.environ["GENOME_SUBSET"] = subset
    mod = types.ModuleType(f"cfg_{arch}_{subset}")
    with open(cfg_path) as f:
        exec(compile(f.read(), str(cfg_path), 'exec'), mod.__dict__)
    r = {
        "arch": arch,
        "subset": subset,
        "batch": mod.batch_size,
        "grad_accum": mod.gradient_accumulation_steps,
        "n_gpus": 4,  # production spec
        "global_batch": mod.batch_size * mod.gradient_accumulation_steps * 4,
        "learning_rate": mod.learning_rate,
        "dataset_dir": mod.dataset_dir,
        "early_stopping": getattr(mod, "early_stopping", None),
    }
    if arch == "bert":
        r["max_iters"] = mod.max_steps
        r["warmup"] = mod.warmup_steps
    else:
        r["max_iters"] = mod.max_iters
        r["warmup"] = mod.warmup_iters
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets-file", required=True)
    ap.add_argument("--learning-source", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    os.environ["LEARNING_SOURCE_DIR"] = args.learning_source

    subsets = [ln.strip() for ln in Path(args.subsets_file).read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]

    rows: list[dict] = []
    errs: list[tuple[str, str, str]] = []
    for subset in subsets:
        for arch, cfg in (("bert", "bert_small_subset.py"),
                          ("gpt2", "gpt2_small_subset.py")):
            cfg_path = ROOT / "molcrawl" / "tasks" / "pretrain" / "configs" / "genome_sequence" / cfg
            try:
                r = evalcfg(cfg_path, arch, subset)
                rows.append(r)
            except Exception as e:
                errs.append((arch, subset, f"{type(e).__name__}: {e}"))

    # Sanity checks: all global_batch == 2560, all early_stopping == False
    n_ok_gb = sum(1 for r in rows if r["global_batch"] == 2560)
    n_ok_es = sum(1 for r in rows if r["early_stopping"] is False)

    lines = [
        "# 42 subset config resolved 値 (readiness 用オフライン検証)",
        "",
        f"検証時刻: {os.popen('date -Iseconds').read().strip()}",
        f"検証対象: 21 subset × 2 arch = 42 config",
        f"eval 成功: **{len(rows)}/42**  |  eval エラー: {len(errs)}",
        "",
        f"- global_batch == 2560: **{n_ok_gb}/{len(rows)}** {'✅' if n_ok_gb == len(rows) else '⚠️'}",
        f"- early_stopping == False: **{n_ok_es}/{len(rows)}** {'✅' if n_ok_es == len(rows) else '⚠️'}",
        "",
        "## 全 config resolved 値",
        "",
        "| arch | subset | batch | grad_accum | global_batch | LR | max_iters | warmup | early_stop |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arch']} | {r['subset']} | {r['batch']} | {r['grad_accum']} | "
            f"{r['global_batch']:,} | {r['learning_rate']} | {r['max_iters']:,} | "
            f"{r['warmup']:,} | {r['early_stopping']} |"
        )
    if errs:
        lines.append("")
        lines.append("## eval エラー")
        lines.append("")
        for arch, subset, msg in errs:
            lines.append(f"- **{arch} / {subset}**: {msg}")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print(f"[OK] wrote {args.out_md} ({len(rows)}/42 rows, {len(errs)} errors)")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
