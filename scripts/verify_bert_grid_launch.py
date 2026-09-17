"""Check the 32 BERT runs of all-bert-order-2026-09-17 against §6 before they are submitted.

Every config is executed the way main.py executes it, in its own process with that
modality's input and output roots, and the resolved values are compared with what the
order requires. The two fp32/bf16 check runs are the rna small base with the command-line
overrides their launcher passes, applied in the same order.

Output directories are the ones the launchers will use: molecule_nat_lang's launcher
derives its own (RUNS_ROOT/bert-output-shuffled/<config>), the others take the config's
model_path. Each is checked for existing checkpoints, since SEGMENT=1 refuses those.

    python scripts/verify_bert_grid_launch.py --main <main checkout>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

C = "molcrawl/tasks/pretrain/configs"
GRIDS = {
    "molecule_nat_lang": [
        f"{C}/molecule_nat_lang/bert_{s}_lr{r}.py" for s in ("small", "medium", "large") for r in ("1e4", "3e4", "1e3")
    ],
    "compounds": [f"{C}/compounds/bert_small_lr{r}.py" for r in ("5e4", "1e3", "2e3")],
    "rna": [f"{C}/rna/bert_{s}_lr{r}.py" for s in ("small", "medium", "large") for r in ("1e4", "3e4", "1e3")],
    "protein_sequence": [
        f"{C}/protein_sequence/bert_{s}_lr{r}.py" for s in ("small", "medium", "large") for r in ("1e4", "3e4", "1e3")
    ],
}
EXPECT_STEPS = {
    "molecule_nat_lang": (12000, 1200),
    "compounds": (15000, 1500),
    "rna": (120960, 12096),
    "protein_sequence": (33531, 3353),
}

CHILD = r"""
import json, runpy, sys
g = runpy.run_path(sys.argv[1], run_name="__main__")
over = json.loads(sys.argv[2])
g.update(over)
keys = ("bf16","dataloader_num_workers","dataloader_pin_memory","batch_size","gradient_accumulation_steps",
        "expected_global_batch","adam_beta2","max_steps","warmup_steps","seed","model_path","learning_rate","stop_at_step")
print(json.dumps({k: g.get(k) for k in keys}, default=str))
"""


def roots(main):
    return {
        "molecule_nat_lang": (f"{main}/learning_source", f"{main}/learning_source_20260803_molnl_train"),
        "compounds": (f"{main}/learning_source_20260805_compounds_packed", f"{main}/learning_source_compounds_runs"),
        "rna": (f"{main}/learning_source_20260723_b200prep", f"{main}/learning_source_rna_runs"),
        "protein_sequence": (f"{main}/learning_source_20260730_protein_uniref50", f"{main}/learning_source_protein_runs"),
    }


def resolve(main, modality, cfg, overrides=None):
    lsd, out_root = roots(main)[modality]
    env = dict(
        os.environ,
        LEARNING_SOURCE_DIR=lsd,
        MODEL_OUTPUT_ROOT=out_root,
        GPT2_TOKENIZER_DIR=f"{main}/assets/tokenizers/gpt2",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=os.getcwd(),
    )
    r = subprocess.run([sys.executable, "-c", CHILD, cfg, json.dumps(overrides or {})], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip().splitlines()[-1] if r.stderr else "failed"
    v = json.loads(r.stdout.strip().splitlines()[-1])
    if modality == "molecule_nat_lang":
        v["model_path"] = f"{out_root}/molecule_nat_lang/bert-output-shuffled/{os.path.basename(cfg)[:-3]}"
    return v, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", required=True)
    a = ap.parse_args()

    runs = [(m, c, None) for m, cs in GRIDS.items() for c in cs]
    for prec, bf in (("fp32", False), ("bf16", True)):
        runs.append(
            (
                "rna",
                f"{C}/rna/bert_small.py",
                {
                    "max_steps": 40320,
                    "warmup_steps": 4032,
                    "stop_at_step": 8000,
                    "bf16": bf,
                    "model_path": f"{roots(a.main)['rna'][1]}/rna/bert-precision-check/bert_small-{prec}",
                },
            )
        )

    fails, outs, seeds = [], {}, {}
    print(
        f"{'#':>2} {'config':34s}{'bf16':>6}{'wk':>3}{'pin':>5}{'eb':>6}{'decl':>6}{'beta2':>7}{'max':>8}{'warm':>7}{'seed':>6}{'stop':>6}  ckpts  model_path"
    )
    for i, (m, cfg, over) in enumerate(runs, 1):
        v, err = resolve(a.main, m, cfg, over)
        name = os.path.basename(cfg)[:-3] + ("" if not over else f" [{('bf16' if over['bf16'] else 'fp32')}]")
        if v is None:
            print(f"{i:>2} {name:34s} FAILED TO EXECUTE: {err}")
            fails.append((name, "execute"))
            continue
        eb = int(v["batch_size"]) * int(v["gradient_accumulation_steps"]) * 4
        ck = len(glob.glob(os.path.join(v["model_path"], "checkpoint-*")))
        print(
            f"{i:>2} {name:34s}{str(v['bf16']):>6}{v['dataloader_num_workers']:>3}{str(v['dataloader_pin_memory']):>5}{eb:>6}"
            f"{str(v['expected_global_batch']):>6}{str(v['adam_beta2']):>7}{v['max_steps']:>8}{v['warmup_steps']:>7}{v['seed']:>6}{v.get('stop_at_step') or 0:>6}  {ck:>5}  {v['model_path']}"
        )
        want_bf16 = over["bf16"] if over else True
        checks = {
            "1 bf16": v["bf16"] is want_bf16,
            "2 workers/pin": v["dataloader_num_workers"] == 4 and v["dataloader_pin_memory"] is True,
            "3 eb 2560": eb == 2560,
            "4 declared 2560": v["expected_global_batch"] == 2560,
            "5 protein beta2": m != "protein_sequence" or v["adam_beta2"] == 0.999,
            "6/7 schedule": (v["max_steps"], v["warmup_steps"]) == ((40320, 4032) if over else EXPECT_STEPS[m]),
            "10 no checkpoints": ck == 0,
        }
        for k, ok in checks.items():
            if not ok:
                fails.append((name, k))
        outs.setdefault(v["model_path"], []).append(name)
        seeds.setdefault((m, "check" if over else "grid"), set()).add(v["seed"])
    for p, names in outs.items():
        if len(names) > 1:
            fails.append((",".join(names), f"9 shared output {p}"))
    for key, s in seeds.items():
        if len(s) > 1:
            fails.append((str(key), f"8 seeds differ {sorted(s)}"))
    print(f"\n{len(runs)} runs checked; {len(outs)} distinct output directories")
    print("seeds per group:", {f"{k[0]}/{k[1]}": sorted(v) for k, v in seeds.items()})
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  ", f)
        return 1
    print("all §6 checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
