#!/usr/bin/env python3
"""The parameter counts and the tokens per step a compute axis is built from.

C = 6ND needs N to be the non-embedding parameter count -- embeddings barely enter the
matrix multiplies, and mol_nl's 50,257-token vocabulary puts tens of millions into them --
and D to be the tokens actually processed. Both are read from what the runs wrote rather
than recomputed from a config: the safetensors header for an HF checkpoint, the saved
state dict for a nanoGPT one, and each run's own run_manifest.json for the batch shape.

    python scripts/molnl_model_sizes.py --bert-root <dir> --gpt2-root <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct

PF_DAY = 8.64e19


def _header(path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def hf_params(ckpt):
    """(total, non-embedding) from the safetensors headers. No weights are read.

    A model past a few GB is saved in shards with an index beside them, which is how the
    xl checkpoints are written; reading only model.safetensors found nothing there.
    """
    single = os.path.join(ckpt, "model.safetensors")
    if os.path.exists(single):
        shards = [single]
    else:
        index = os.path.join(ckpt, "model.safetensors.index.json")
        if not os.path.exists(index):
            raise SystemExit(f"{ckpt}: neither model.safetensors nor an index beside it")
        names = sorted(set(json.load(open(index))["weight_map"].values()))
        shards = [os.path.join(ckpt, n) for n in names]
    total = 0
    seen = set()
    for shard in shards:
        for name, meta in _header(shard).items():
            if name == "__metadata__" or name in seen:
                continue
            seen.add(name)
            size = 1
            for dim in meta["shape"]:
                size *= dim
            total += size
    conf = json.load(open(os.path.join(ckpt, "config.json")))
    h, v = conf["hidden_size"], conf["vocab_size"]
    pos = conf.get("max_position_embeddings", 0)
    return total, total - (v * h + pos * h + 2 * h), conf


def nanogpt_params(path):
    """(total, non-embedding) from the checkpoint's own tensors.

    nanoGPT ties the language-model head to the token embedding but saves both tensors,
    so summing the state dict counts the embedding matrix twice: GPT-2 small reads
    162,935,040 where the model holds 124,337,664. The duplicate is dropped here after
    checking that the two really do carry the same values, and the embeddings come off
    on top of that -- what is left is the N that 6ND wants.
    """
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    wte = sd.get("transformer.wte.weight")
    head = sd.get("lm_head.weight")
    tied = (wte is not None and head is not None and wte.shape == head.shape
            and bool((wte == head).all()))
    total = sum(t.numel() for k, t in sd.items()
                if not (tied and k == "lm_head.weight"))
    embed = sum(t.numel() for k, t in sd.items()
                if k in ("transformer.wte.weight", "transformer.wpe.weight"))
    if not tied and head is not None:
        raise SystemExit(f"{path}: lm_head is not tied to wte; decide what N means first")
    return total, total - embed, ck.get("model_args", {})


def batch_shape(run_dir):
    """tokens per step, and the two numbers it is the product of.

    A table that states only the product cannot be checked: 2,621,440 is as consistent
    with 2,560 sequences of 1,024 as with 1,280 of 2,048. The 2026-11-12 order asks for
    both, so both come out of the manifest the run wrote.
    """
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(path):
        return {}
    batch = json.load(open(path)).get("batch", {})
    sequences = batch.get("effective_global_batch") or batch.get("global_batch")
    seq_len = batch.get("seq_len", batch.get("max_length"))
    tokens = batch.get("tokens_per_step")
    if tokens is None and sequences and seq_len:
        tokens = sequences * seq_len
    return {"tokens_per_step": tokens,
            "global_batch_sequences": sequences,
            "sequence_length": seq_len}


def tokens_per_step(run_dir):
    return batch_shape(run_dir).get("tokens_per_step")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bert-root", required=True)
    ap.add_argument("--gpt2-root", required=True)
    ap.add_argument("--bert-runs", nargs="+", required=True)
    ap.add_argument("--gpt2-runs", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    out = {}
    print(f"{'arch':<5}{'run':<28}{'総数':>14}{'非埋め込み N':>15}{'tokens/step':>13}"
          f"{'層':>5}{'幅':>6}")
    for name in a.bert_runs:
        run = os.path.join(a.bert_root, name)
        cks = sorted(glob.glob(os.path.join(run, "checkpoint-*")),
                     key=lambda p: int(p.rsplit("-", 1)[1]))
        total, nonembed, conf = hf_params(cks[-1])
        shape = batch_shape(run)
        tps = shape.get("tokens_per_step")
        out[name] = {"arch": "bert", "total": total, "non_embedding": nonembed,
                     "non_embedding_rule": ("total - (vocab + max_position) * hidden - 2 * "
                                            "hidden; the same subtraction as "
                                            "scripts/compounds_ladder_table.bert_nonembedding"),
                     "layers": conf["num_hidden_layers"],
                     "hidden": conf["hidden_size"], "vocab": conf["vocab_size"],
                     "checkpoint": os.path.basename(cks[-1]), **shape}
        print(f"{'bert':<5}{name:<28}{total:>14,}{nonembed:>15,}{tps:>13,}"
              f"{conf['num_hidden_layers']:>5}{conf['hidden_size']:>6}")
    for name in a.gpt2_runs:
        run = os.path.join(a.gpt2_root, name)
        total, nonembed, margs = nanogpt_params(os.path.join(run, "ckpt.pt"))
        shape = batch_shape(run)
        tps = shape.get("tokens_per_step")
        # Two subtractions are in use in this repository and they disagree for a tied
        # GPT-2. This file takes wte and wpe off (Kaplan's N, what molnl's compute axis
        # has always used); scripts/compounds_ladder_table.gpt2_nonembedding takes only
        # wpe off, on the ground that wte is tied to lm_head and so its parameters do
        # enter the output matrix multiply. The 2026-11-12 order asks for the compounds
        # derivation, so both are written and named rather than one being silently
        # swapped for the other under the same key.
        wte = (margs.get("vocab_size") or 0) * (margs.get("n_embd") or 0)
        out[name] = {"arch": "gpt2", "total": total, "non_embedding": nonembed,
                     "non_embedding_rule": ("total - wte - wpe (Kaplan's N; what molnl's "
                                            "PF-days axis uses)"),
                     "non_embedding_compounds_rule": nonembed + wte,
                     "non_embedding_compounds_rule_note": ("total - wpe only; wte is tied "
                                                           "to lm_head and enters the "
                                                           "output matmul. Matches "
                                                           "scripts/compounds_ladder_table"
                                                           ".gpt2_nonembedding"),
                     "layers": margs.get("n_layer"),
                     "hidden": margs.get("n_embd"), "vocab": margs.get("vocab_size"),
                     "checkpoint": "ckpt.pt", **shape}
        print(f"{'gpt2':<5}{name:<28}{total:>14,}{nonembed:>15,}{tps:>13,}"
              f"{margs.get('n_layer', 0):>5}{margs.get('n_embd', 0):>6}")
    print(f"\n  1 PF-day = {PF_DAY:.3g} FLOP. C = 6ND omits the attention term, "
          f"13-18% at sequence length 1,024.")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
