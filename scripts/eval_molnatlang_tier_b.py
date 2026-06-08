"""Tier B verification — per-token & per-position cross-entropy loss diff (molnatlang).

Adapted from scripts/eval_protein_tier_b.py for molecule_nat_lang × gpt2-small.
The molnatlang tokenizer is the OpenAI GPT-2 byte-pair encoder (vocab=50257),
so per-token aggregation is dominated by extremely long tail. The script
reports top-K by Δ (improvement / regression) and top-K by count.
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
        targets = ids[:, 1:]
        logits, _ = model(ids[:, :-1], targets=targets)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        loss = loss.view(targets.shape)

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


def load_molnatlang_vocab() -> dict[int, str]:
    from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

    tok = MoleculeNatLangTokenizer()
    # MoleculeNatLangTokenizer wraps a HF GPT-2 tokenizer at tok.tokenizer
    return {i: tok.tokenizer.decode([i]) for i in range(tok.vocab_size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_ckpt", required=True)
    ap.add_argument("--new_ckpt", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--out_json", default="tier_b_molnatlang_results.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Loading dataset {args.dataset_dir}/{args.split}...", flush=True)
    ds_dict = load_from_disk(args.dataset_dir)
    ds = ds_dict[args.split]
    print(f"  → {len(ds)} chunks of length {len(ds[0]['input_ids'])}", flush=True)

    print("Loading GPT-2 tokenizer vocab via MoleculeNatLangTokenizer...", flush=True)
    vocab = load_molnatlang_vocab()
    print(f"  → vocab_size={len(vocab)}", flush=True)

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

    # Per-token diffs
    rows = []
    token_ids = set(results["old"]["per_token"].keys()) | set(results["new"]["per_token"].keys())
    for tid in token_ids:
        o = results["old"]["per_token"].get(tid, {"avg": float("nan"), "n": 0})
        nn = results["new"]["per_token"].get(tid, {"avg": float("nan"), "n": 0})
        if not o["n"] or not nn["n"]:
            continue
        delta = o["avg"] - nn["avg"]
        ratio = o["avg"] / nn["avg"] if nn["avg"] > 0 else float("nan")
        piece = vocab.get(tid, f"<{tid}>").replace("\n", "\\n").replace("\t", "\\t")
        rows.append((tid, piece, max(o["n"], nn["n"]), o["avg"], nn["avg"], delta, ratio))

    print(f"\n########## top-{args.top_k} BIGGEST IMPROVEMENT (Δ = old - new) ##########")
    print(f"{'tok_id':>6}  {'piece':>16}  {'count':>8}  {'old_loss':>10}  {'new_loss':>10}  {'Δ':>10}  {'×':>6}")
    for tid, piece, n, ol, nl, d, r in sorted(rows, key=lambda r: -r[5])[: args.top_k]:
        print(f"{tid:>6}  {piece[:16]:>16}  {n:>8}  {ol:>10.4f}  {nl:>10.4f}  {d:>10.4f}  {r:>6.2f}")

    print(f"\n########## top-{args.top_k} BIGGEST REGRESSION (Δ < 0 = new is worse) ##########")
    print(f"{'tok_id':>6}  {'piece':>16}  {'count':>8}  {'old_loss':>10}  {'new_loss':>10}  {'Δ':>10}  {'×':>6}")
    for tid, piece, n, ol, nl, d, r in sorted(rows, key=lambda r: r[5])[: args.top_k]:
        print(f"{tid:>6}  {piece[:16]:>16}  {n:>8}  {ol:>10.4f}  {nl:>10.4f}  {d:>10.4f}  {r:>6.2f}")

    print(f"\n########## top-{args.top_k} BY COUNT (high-frequency pieces) ##########")
    print(f"{'tok_id':>6}  {'piece':>16}  {'count':>8}  {'old_loss':>10}  {'new_loss':>10}  {'Δ':>10}  {'×':>6}")
    for tid, piece, n, ol, nl, d, r in sorted(rows, key=lambda r: -r[2])[: args.top_k]:
        print(f"{tid:>6}  {piece[:16]:>16}  {n:>8}  {ol:>10.4f}  {nl:>10.4f}  {d:>10.4f}  {r:>6.2f}")

    # Per-position summary
    import numpy as np

    print("\n\n########## per-position loss summary ##########")
    old_pos = np.asarray(results["old"]["per_position_avg_loss"])
    new_pos = np.asarray(results["new"]["per_position_avg_loss"])
    edges = [0, 50, 200, 500, 1023]
    print(f"{'positions':>14}  {'old_avg':>10}  {'new_avg':>10}  {'Δ':>10}")
    for a, b in zip(edges[:-1], edges[1:]):
        oa = float(old_pos[a:b].mean())
        na = float(new_pos[a:b].mean())
        print(f"  [{a:>4}, {b:>4})  {oa:>10.4f}  {na:>10.4f}  {oa - na:>10.4f}")
    oa = float(old_pos.mean())
    na = float(new_pos.mean())
    print(f"     overall    {oa:>10.4f}  {na:>10.4f}  {oa - na:>10.4f}")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
