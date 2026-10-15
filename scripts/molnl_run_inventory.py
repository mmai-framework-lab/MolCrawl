#!/usr/bin/env python3
"""What each run was, and whether it could be picked up again.

Written for a maintenance window: afterwards the run directories are still there but the
cluster is not, so the facts that live in run_manifest.json and in the checkpoints have to
be on paper before the machine goes away.

Per run: the conditions the run resolved to, the newest checkpoint, and whether a
continuation would be allowed to start from it -- the ladder script refuses a resume whose
config disagrees with the checkpoint's own max_steps, and that check is reproduced here
rather than guessed at.

    python scripts/molnl_run_inventory.py --runs-root <bert-output-shuffled> --out <file.tsv>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

COLUMNS = ["run", "job", "state_hint", "seed", "learning_rate", "max_steps", "warmup_steps",
           "per_device", "grad_accum", "world_size", "effective_global_batch",
           "fixed_eval_mask", "bf16", "eval_rows", "newest_checkpoint", "checkpoint_max_steps",
           "resumable", "git"]


def newest_checkpoint(run_dir):
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    return cks[-1] if cks else None


def row_for(run_dir):
    name = os.path.basename(run_dir)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    m = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    batch, schedule = m.get("batch", {}), m.get("schedule", {})
    evaluation, placement = m.get("eval", {}), m.get("placement", {})
    ck = newest_checkpoint(run_dir)
    ck_steps, done, resumable = "", "", "no checkpoint"
    if ck:
        state = json.load(open(os.path.join(ck, "trainer_state.json")))
        ck_steps = int(state.get("max_steps") or 0)
        done = int(state.get("global_step") or 0)
        # The ladder script allows a continuation when the checkpoint's own max_steps
        # matches the config's and the run has not already reached it.
        if ck_steps and done >= ck_steps:
            resumable = "finished"
        elif ck_steps and ck_steps == (schedule.get("max_steps") or ck_steps):
            resumable = f"yes, from step {done:,} of {ck_steps:,}"
        else:
            resumable = "schedule mismatch"
    job = (m.get("run", {}) or {}).get("job_id", "")
    if not job:
        log = os.path.join(run_dir, "segments.log")
        if os.path.exists(log):
            for line in open(log):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    job = parts[1]
    return [
        name, job or "", "", (m.get("seed", {}) or {}).get("seed", ""),
        (m.get("sources", {}).get("learning_rate", {}) or {}).get("value", ""),
        schedule.get("max_steps", ""), schedule.get("warmup_steps", ""),
        batch.get("per_device_train_batch_size", ""), batch.get("gradient_accumulation_steps", ""),
        placement.get("world_size", ""), batch.get("effective_global_batch", ""),
        (evaluation.get("fixed_mask", {}) or {}).get("enabled", ""),
        (m.get("sources", {}).get("bf16", {}) or {}).get("value", ""),
        evaluation.get("rows", ""), os.path.basename(ck) if ck else "", ck_steps, resumable,
        ((m.get("run", {}) or {}).get("git", {}) or {}).get("commit", "")[:12],
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--glob", default="bert_*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [row_for(d) for d in sorted(glob.glob(os.path.join(a.runs_root, a.glob)))
            if os.path.isdir(d)]
    with open(a.out, "w") as fh:
        fh.write("# molecule_nat_lang BERT runs: the conditions each resolved to, and\n")
        fh.write("# whether a continuation could start from its newest checkpoint\n")
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"  {len(rows)} runs -> {a.out}")
    for row in rows:
        print(f"    {row[0]:<40}{str(row[14]):<18}{row[16]}")
    missing = [r[0] for r in rows if not r[1]]
    if missing:
        print(f"  job id のない run: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
