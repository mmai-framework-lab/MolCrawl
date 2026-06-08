"""Tier B verification — per-token & per-position cross-entropy loss diff.

Compares the legacy 50k-capped protein_sequence-small (OLD) and the bugfix
5M-cap retrain (NEW) on a shared held-out chunked test set. For each model:

* aggregates cross-entropy loss bucketed by *target token id*
  → highlights rare/ambiguous amino acids (U, O, X, B, J, Z) where the OLD
    model saw effectively zero training examples
* aggregates cross-entropy loss bucketed by *within-chunk position* (0..1022)
  → shows whether the long-context tail is what improved (NEW had ~100x more
    long-sequence material reaching the chunk pipeline)

Outputs a JSON dump + a human-readable comparison table on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk

# Repo-root imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from molcrawl.models.gpt2.model import GPT, GPTConfig  # noqa: E402


def load_ckpt(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    args = ckpt["model_args"]
    model = GPT(GPTConfig(**args))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod.") :]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, args


@torch.no_grad()
def accumulate(model, dataset, max_samples: int, device: str, block_size: int = 1024):
    sum_per_token: dict[int, float] = defaultdict(float)
    cnt_per_token: dict[int, int] = defaultdict(int)
    pos_sum = torch.zeros(block_size - 1, dtype=torch.float64, device=device)
    pos_cnt = 0

    n = min(max_samples, len(dataset))
    for i in range(n):
        row = dataset[i]
        ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        # GPT.forward(idx, targets=None) returns logits only for the LAST position
        # when targets is None (nanoGPT inference optimisation). Passing targets
        # makes it compute logits for every position, which is what we need here.
        targets = ids[:, 1:]  # (1, T-1)
        logits, _ = model(ids[:, :-1], targets=targets)  # logits: (1, T-1, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )  # (T-1,)
        loss = loss.view(targets.shape)  # (1, T-1)

        tgt_flat = targets[0].tolist()
        loss_flat = loss[0].tolist()
        for t, l in zip(tgt_flat, loss_flat):
            sum_per_token[int(t)] += float(l)
            cnt_per_token[int(t)] += 1

        pos_sum += loss[0].to(torch.float64)
        pos_cnt += 1

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{n} samples", flush=True)

    return sum_per_token, cnt_per_token, pos_sum.cpu().numpy(), pos_cnt


def load_tokenizer_vocab():
    from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer

    tok = EsmSequenceTokenizer()
    inv = {tok.convert_tokens_to_ids(t): t for t in tok.get_vocab().keys()}
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_ckpt", required=True)
    ap.add_argument("--new_ckpt", required=True)
    ap.add_argument("--dataset_dir", required=True, help="HF dataset with train/valid/test splits")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--out_json", default="tier_b_results.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Loading dataset {args.dataset_dir}/{args.split}...", flush=True)
    ds_dict = load_from_disk(args.dataset_dir)
    ds = ds_dict[args.split]
    print(f"  → {len(ds)} chunks of length {len(ds[0]['input_ids'])}", flush=True)

    print("Loading tokenizer vocab...", flush=True)
    vocab = load_tokenizer_vocab()

    results: dict[str, object] = {"dataset_dir": args.dataset_dir, "split": args.split, "max_samples": args.max_samples}

    for tag, ckpt_path in [("old", args.old_ckpt), ("new", args.new_ckpt)]:
        print(f"\n========== {tag.upper()} ckpt: {ckpt_path} ==========", flush=True)
        model, model_args = load_ckpt(ckpt_path, args.device)
        print(f"  vocab_size={model_args['vocab_size']} block_size={model_args['block_size']}", flush=True)
        s, c, ps, pc = accumulate(model, ds, args.max_samples, args.device)
        per_token = {int(k): {"sum": s[k], "n": c[k], "avg": s[k] / c[k]} for k in s}
        per_pos_avg = (ps / pc).tolist() if pc > 0 else []
        results[tag] = {"per_token": per_token, "per_position_avg_loss": per_pos_avg}
        del model
        torch.cuda.empty_cache()

    # Diff & report
    print("\n\n########## per-token loss comparison ##########")
    print(f"{'tok_id':>6}  {'token':>10}  {'count':>8}  {'old_loss':>10}  {'new_loss':>10}  {'Δ (old−new)':>13}  {'×':>6}")
    token_ids = sorted(set(results["old"]["per_token"].keys()) | set(results["new"]["per_token"].keys()))
    rows = []
    for tid in token_ids:
        o = results["old"]["per_token"].get(tid, {"avg": float("nan"), "n": 0})
        nn = results["new"]["per_token"].get(tid, {"avg": float("nan"), "n": 0})
        delta = o["avg"] - nn["avg"] if o["n"] and nn["n"] else float("nan")
        ratio = o["avg"] / nn["avg"] if (o["n"] and nn["n"] and nn["avg"] > 0) else float("nan")
        name = vocab.get(tid, f"<{tid}>")
        rows.append((tid, name, max(o["n"], nn["n"]), o["avg"], nn["avg"], delta, ratio))
    # Sort by reduction
    rows_sorted = sorted(rows, key=lambda r: -r[5] if r[5] == r[5] else float("inf"))
    for tid, name, n, ol, nl, d, r in rows_sorted:
        print(f"{tid:>6}  {name:>10}  {n:>8}  {ol:>10.4f}  {nl:>10.4f}  {d:>13.4f}  {r:>6.2f}")

    print("\n\n########## per-position loss summary ##########")
    import numpy as np

    old_pos = np.asarray(results["old"]["per_position_avg_loss"])
    new_pos = np.asarray(results["new"]["per_position_avg_loss"])
    edges = [0, 50, 200, 500, 1023]
    print(f"{'positions':>14}  {'old_avg':>10}  {'new_avg':>10}  {'Δ':>10}")
    for a, b in zip(edges[:-1], edges[1:]):
        oa = float(old_pos[a:b].mean())
        na = float(new_pos[a:b].mean())
        print(f"  [{a:>4}, {b:>4})  {oa:>10.4f}  {na:>10.4f}  {oa - na:>10.4f}")
    # Whole-curve summary
    oa = float(old_pos.mean())
    na = float(new_pos.mean())
    print(f"     overall    {oa:>10.4f}  {na:>10.4f}  {oa - na:>10.4f}")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
