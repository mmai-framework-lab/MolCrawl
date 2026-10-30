"""§2 diagnostic: does the protein GPT-2 exploit cross-protein (co-residence) context?

Runs one eval pass over the valid split of a packed protein GPT-2 dataset and splits
the per-token loss two ways:
  - group 1: the token's protein is the FIRST in its 1024 block (no prior protein to
    its left inside the block)
  - group 2: the token's protein is the 2nd-or-later in its block (a prior protein +
    EOS sit to its left)
Within each group the loss is bucketed by "how many residues into its own protein" the
target is (k=0 is the first residue of that protein).

Reading (per the instruction): if group-2 loss is much LOWER than group-1 at small k
(the base / early residues), the model is predicting a protein's own early residues
from the neighbouring protein = cross-protein leakage. If the two are close, there is
no meaningful co-residence exploitation. Cost is one eval pass, no training.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F


def main() -> int:
    p = ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="valid")
    p.add_argument("--eos-id", type=int, default=2)
    p.add_argument("--n-blocks", type=int, default=3000)
    p.add_argument("--batch-blocks", type=int, default=16)
    p.add_argument("--max-k", type=int, default=64, help="report per-position loss for k in [0, max-k)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from datasets import load_from_disk

    from molcrawl.models.gpt2.model import GPT, GPTConfig

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    conf = GPTConfig(**ck["model_args"])
    model = GPT(conf)
    sd = {(k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.eval().to(args.device)
    T = ck["model_args"]["block_size"]
    eos = args.eos_id
    MAXK = args.max_k

    ds = load_from_disk(args.dataset_dir)[args.split]
    n = min(args.n_blocks, len(ds))

    # sum_loss[group, k], cnt[group, k]  (group 0 = first protein in block, 1 = 2nd+)
    sum_loss = np.zeros((2, MAXK), dtype=np.float64)
    cnt = np.zeros((2, MAXK), dtype=np.int64)
    # also overall group means (all k, excluding EOS targets)
    g_sum = np.zeros(2, dtype=np.float64)
    g_cnt = np.zeros(2, dtype=np.int64)

    torch.manual_seed(0)
    with torch.no_grad():
        for start in range(0, n, args.batch_blocks):
            rows = [ds[i]["input_ids"] for i in range(start, min(start + args.batch_blocks, n))]
            idx = torch.tensor(rows, dtype=torch.long, device=args.device)  # (B, T)
            logits, _ = model(idx, targets=idx)  # targets given -> full-position logits
            # per-token CE for predicting position i+1 from context 0..i
            lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)  # (B, T-1, V)
            tgt = idx[:, 1:]  # (B, T-1)
            tok_loss = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
            tok_loss = tok_loss.cpu().numpy()
            blk = idx.cpu().numpy()  # (B, T)
            for b in range(blk.shape[0]):
                toks = blk[b]
                # segment id of each token = number of EOS strictly before it
                is_eos = (toks == eos).astype(np.int64)
                seg = np.concatenate([[0], np.cumsum(is_eos)[:-1]])  # seg[i]
                # start index of each token's segment
                seg_start = np.zeros(T, dtype=np.int64)
                cur = 0
                for i in range(T):
                    if i > 0 and toks[i - 1] == eos:
                        cur = i
                    seg_start[i] = cur
                within = np.arange(T) - seg_start  # position within own protein
                # assign loss[i] (predicting target at i+1) to target's (group,k)
                for i in range(T - 1):
                    j = i + 1
                    if toks[j] == eos:
                        continue  # skip EOS targets
                    grp = 0 if seg[j] == 0 else 1
                    k = within[j]
                    L = tok_loss[b, i]
                    g_sum[grp] += L
                    g_cnt[grp] += 1
                    if k < MAXK:
                        sum_loss[grp, k] += L
                        cnt[grp, k] += 1

    g1 = sum_loss[0] / np.maximum(cnt[0], 1)
    g2 = sum_loss[1] / np.maximum(cnt[1], 1)
    base = slice(0, 8)  # k in [0,8) = the "base" / early residues
    base_diff = float(np.nanmean(g1[base]) - np.nanmean(g2[base]))  # >0 means group2 lower = leakage
    out = {
        "checkpoint": args.checkpoint,
        "n_blocks": n,
        "group_mean_all_k": {"first_protein": g_sum[0] / max(g_cnt[0], 1), "second_plus": g_sum[1] / max(g_cnt[1], 1)},
        "base_k0_7_first": float(np.nanmean(g1[base])),
        "base_k0_7_secondplus": float(np.nanmean(g2[base])),
        "base_diff_first_minus_secondplus": base_diff,
        "note": ">0 means 2nd+ protein predicts its own early residues better = cross-protein leakage",
    }
    print("RESULT " + json.dumps(out))
    print("k    first_protein   second_plus   (first - second)")
    for k in range(min(MAXK, 24)):
        print(f"{k:>3}  {g1[k]:>12.4f}  {g2[k]:>12.4f}   {g1[k] - g2[k]:>+8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
