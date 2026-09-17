"""Collect the run facts all-bert-order-2026-09-17 asks to be reported, from the runs themselves.

Three questions, each answered from what a run left on disk rather than from what a
config says today -- configs move after a run has finished:

  genome     which precision and warmup the genome BERT runs trained with. Read from
             each run's saved TrainingArguments (training_args.bin in its newest
             checkpoint), which is what Trainer actually used.
  compounds  the effective global batch of the compounds GPT-2 runs, from their
             run_manifest.json where one exists and from the nanoGPT checkpoint's
             saved config otherwise.
  superseded the fields of the stopped molecule_nat_lang runs' run_manifest.json.

    python scripts/collect_bert_order_facts.py --genome-runs <dir> \
        --compounds-root <learning_source_..._compounds_packed> --superseded <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re


def _newest_checkpoint(run_dir):
    cks = [c for c in glob.glob(os.path.join(run_dir, "checkpoint-*")) if re.search(r"checkpoint-\d+$", c)]
    return max(cks, key=lambda c: int(c.rsplit("-", 1)[1])) if cks else None


def genome(runs_root):
    import torch

    print("## genome BERT: TrainingArguments saved in each run's newest checkpoint")
    print(f"{'run':44s}{'ckpt':>8}{'bf16':>6}{'max_steps':>10}{'warmup':>8}{'ratio':>7}{'mb':>5}{'acc':>5}{'epoch@end':>10}")
    for d in sorted(glob.glob(os.path.join(runs_root, "bert-small-*"))):
        ck = _newest_checkpoint(d)
        if not ck or not os.path.exists(os.path.join(ck, "training_args.bin")):
            continue
        a = torch.load(os.path.join(ck, "training_args.bin"), weights_only=False)
        st = json.load(open(os.path.join(ck, "trainer_state.json")))
        ratio = a.warmup_steps / a.max_steps if a.max_steps else float("nan")
        print(
            f"{os.path.basename(d)[len('bert-small-') :]:44s}{int(ck.rsplit('-', 1)[1]):>8}{str(a.bf16):>6}"
            f"{a.max_steps:>10}{a.warmup_steps:>8}{ratio:>7.2%}{a.per_device_train_batch_size:>5}"
            f"{a.gradient_accumulation_steps:>5}{st.get('epoch', float('nan')):>10.2f}"
        )


def compounds(root):
    import torch

    print("\n## compounds GPT-2: effective global batch")
    print(f"{'run':52s}{'source':>10}{'mb':>5}{'acc(cfg)':>9}{'world':>6}{'effective':>10}")
    for m in sorted(glob.glob(os.path.join(root, "runs", "*gpt2*", "run_manifest.json"))):
        b = json.load(open(m)).get("batch", {})
        print(
            f"{os.path.basename(os.path.dirname(m)):52s}{'manifest':>10}{str(b.get('batch_size', b.get('per_device_train_batch_size'))):>5}"
            f"{str(b.get('gradient_accumulation_steps_configured', b.get('gradient_accumulation_steps'))):>9}"
            f"{str(b.get('world_size')):>6}{str(b.get('effective_global_batch')):>10}"
        )
    for d in sorted(glob.glob(os.path.join(root, "compounds", "gpt2-output", "*"))):
        ck = os.path.join(d, "ckpt.pt")
        if not os.path.exists(ck):
            continue
        cfg = torch.load(ck, map_location="cpu", weights_only=False).get("config", {})
        mb, acc = cfg.get("batch_size"), cfg.get("gradient_accumulation_steps")
        eff = mb * acc if isinstance(mb, int) and isinstance(acc, int) else None
        print(f"{os.path.basename(d):52s}{'ckpt':>10}{str(mb):>5}{str(acc):>9}{'-':>6}{str(eff):>10}")
    print("nanoGPT divides the configured accumulation by the world size and multiplies it back,")
    print("so batch_size x configured accumulation is the effective batch at any GPU count.")


def superseded(root):
    print("\n## superseded molecule_nat_lang runs: run_manifest.json fields")
    for m in sorted(glob.glob(os.path.join(root, "*", "run_manifest.json"))):
        j = json.load(open(m))
        r, b, s = j.get("run", {}), j.get("batch", {}), j.get("schedule", {})
        print(
            json.dumps(
                {
                    "config_dir": os.path.basename(os.path.dirname(m)),
                    "job_id": r.get("job_id"),
                    "node": r.get("node"),
                    "commit": (r.get("git") or {}).get("commit"),
                    "dirty": (r.get("git") or {}).get("dirty"),
                    "seed": j.get("seed"),
                    "effective_global_batch": b.get("effective_global_batch"),
                    "max_steps": s.get("max_steps"),
                    "warmup_steps": s.get("warmup_steps"),
                    "learning_rate": s.get("learning_rate"),
                    "written": j.get("written"),
                },
                ensure_ascii=False,
            )
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome-runs")
    ap.add_argument("--compounds-root")
    ap.add_argument("--superseded")
    a = ap.parse_args()
    if a.genome_runs:
        genome(a.genome_runs)
    if a.compounds_root:
        compounds(a.compounds_root)
    if a.superseded:
        superseded(a.superseded)


if __name__ == "__main__":
    main()
