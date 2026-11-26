#!/usr/bin/env python3
"""Fine-tune a molnl pretraining checkpoint for ChEBI-20 text-to-molecule retrieval.

The 2026-11-26 order: start from one of the eight runs of item 1, turn each description
and each molecule into a vector, train so that the vectors of a pair are close, then
rank all 33,010 ChEBI-20 molecules for every test description by that closeness. Token
likelihood plays no part.

What is held the same for all eight runs (item 4) is the SETTINGS dict below, written
into run_manifest.json as is. The model is the only thing that differs:

  vector        one encoder -- the pretrained model -- reads both sides. Each side is
                written as the half of a corpus record it would have been in pretraining:
                the molecule as "<s>[INST] <SMILES> [/INST]", the description as
                "<description><|endoftext|>". The vector is the mean of the final
                layer's hidden states over the side's own tokens (GPT-2: after ln_f;
                BERT: last_hidden_state), then a linear map to 256 dimensions, one map
                per side, then L2 normalisation.
  closeness     cosine similarity: the dot product of the normalised vectors.
  loss          symmetric InfoNCE (CLIP): cross-entropy of each description against
                every molecule in the batch and of each molecule against every
                description, averaged. Negatives are the other pairs in the global
                batch, gathered across GPUs. The temperature is learned, starting at
                0.07, the value Text2Mol starts from.
  optimisation  AdamW, encoder 3e-5 and the two maps and the temperature 1e-4 (Text2Mol's
                two rates), weight decay 0.01, linear warmup over 10 % then linear decay
                to 0, gradient clipping at 1.0, global batch 256 pairs (64 x 4 GPUs),
                40 epochs of ChEBI-20's train split with the last partial batch dropped.
  seed          42: the maps' initialisation, the order of the train split per epoch.

The run evaluates on ChEBI-20's valid split every 50 steps and at step 0: the same loss,
computed over the valid pairs in their file order in fixed blocks of 256, so the number
does not depend on the GPU count; plus the MRR of each valid description against the
3,301 valid molecules, as a reading of the task itself. The checkpoint with the lowest
valid loss is the one the test split is ranked with.

The rank of the correct molecule counts every other candidate whose similarity is equal
or higher, so a tie is never resolved in the model's favour. ChEBI-20 has one pair of
CIDs whose SDF gives the same SMILES; their vectors are identical and the tie is real.

    torchrun --standalone --nproc_per_node=4 scripts/molnl_chebi20_retrieval.py \
        --arch bert --arm <name> --init <checkpoint dir | training_state.bin> \
        --expect-step <step> --data <chebi20.jsonl> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dnn
import torch.nn as nn
import torch.nn.functional as F

SETTINGS = {
    "molecule_template": "<s>[INST] {smiles} [/INST]",
    "description_template": "{description}<|endoftext|>",
    "max_tokens_per_side": 512,
    "pooling": "mean of final-layer hidden states over the side's own tokens",
    "projection_dim": 256,
    "projection": "one linear map per side (description, molecule), with bias",
    "normalise": "L2",
    "similarity": "cosine",
    "loss": "symmetric InfoNCE: (CE(text->mol) + CE(mol->text)) / 2",
    "negatives": "the other pairs of the global batch, gathered across GPUs",
    "temperature_init": 0.07,
    "temperature_learned": True,
    "logit_scale_max": 100.0,
    "optimizer": "AdamW betas (0.9, 0.999) eps 1e-8",
    "lr_encoder": 3e-5,
    "lr_head": 1e-4,
    "weight_decay": 0.01,
    "no_weight_decay_on": "biases, LayerNorm weights, the temperature",
    "warmup_fraction": 0.10,
    "schedule": "linear warmup then linear decay to 0",
    "grad_clip": 1.0,
    "global_batch_pairs": 256,
    "world_size": 4,
    "epochs": 40,
    "drop_last": True,
    "precision": "bf16 autocast, fp32 weights and optimizer",
    "gradient_checkpointing": True,
    "eval_every_steps": 50,
    "eval_block_pairs": 256,
    "selection": "lowest valid loss",
    "seed": 42,
}


def log(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- encoders

class GPT2Encoder(nn.Module):
    """nanoGPT up to ln_f. Right padding is harmless: a causal position never sees it."""

    def __init__(self, init_path, expect_step):
        super().__init__()
        from molcrawl.models.gpt2.model import GPT, GPTConfig

        state = torch.load(init_path, map_location="cpu", weights_only=False)
        if expect_step is not None and int(state["iter_num"]) != expect_step:
            raise SystemExit(f"{init_path}: iter_num {state['iter_num']} != {expect_step}")
        args = dict(state["model_args"])
        self.loaded = {"iter_num": int(state["iter_num"]),
                       "val_loss": state.get("val_loss"), "model_args": args}
        cfg = GPTConfig(**{k: args[k] for k in ("n_layer", "n_head", "n_embd", "block_size",
                                                "bias", "vocab_size", "dropout")})
        self.gpt = GPT(cfg)
        sd = {k.removeprefix("_orig_mod."): v for k, v in state["model"].items()}
        self.gpt.load_state_dict(sd, strict=True)
        del state, sd
        self.hidden_size = cfg.n_embd
        self.dropout = cfg.dropout
        self.checkpointing = False

    def forward(self, ids, mask):
        t = self.gpt.transformer
        pos = torch.arange(ids.size(1), device=ids.device)
        x = t.drop(t.wte(ids) + t.wpe(pos))
        for block in t.h:
            if self.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return t.ln_f(x)


class BertEncoder(nn.Module):
    def __init__(self, init_dir, expect_step):
        super().__init__()
        from transformers import BertModel

        state = json.load(open(os.path.join(init_dir, "trainer_state.json")))
        if expect_step is not None and int(state["global_step"]) != expect_step:
            raise SystemExit(f"{init_dir}: global_step {state['global_step']} != {expect_step}")
        self.bert = BertModel.from_pretrained(init_dir, add_pooling_layer=False)
        self.loaded = {"global_step": int(state["global_step"])}
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = {"hidden": self.bert.config.hidden_dropout_prob,
                        "attention": self.bert.config.attention_probs_dropout_prob}

    def forward(self, ids, mask):
        return self.bert(input_ids=ids, attention_mask=mask).last_hidden_state


class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        h, d = encoder.hidden_size, SETTINGS["projection_dim"]
        self.text_proj = nn.Linear(h, d)
        self.mol_proj = nn.Linear(h, d)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / SETTINGS["temperature_init"])))

    def embed(self, ids, mask, side):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.encoder(ids, mask)
        hs = hs.float()
        m = mask.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
        proj = self.text_proj if side == "text" else self.mol_proj
        return F.normalize(proj(pooled), dim=-1)

    def forward(self, ids, mask, n_text):
        """One call for both sides so DDP sees one forward per step."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.encoder(ids, mask)
        hs = hs.float()
        m = mask.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
        t = F.normalize(self.text_proj(pooled[:n_text]), dim=-1)
        g = F.normalize(self.mol_proj(pooled[n_text:]), dim=-1)
        return t, g, self.logit_scale.exp().clamp(max=SETTINGS["logit_scale_max"])


def clip_loss(t, g, scale):
    logits = scale * t @ g.t()
    labels = torch.arange(t.size(0), device=t.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


# ---------------------------------------------------------------- data

def encode_all(tokenizer, texts, max_tokens):
    out, cut = [], 0
    for chunk in range(0, len(texts), 4096):
        enc = tokenizer(texts[chunk:chunk + 4096], add_special_tokens=False)["input_ids"]
        for ids in enc:
            if len(ids) > max_tokens:
                cut += 1
                ids = ids[:max_tokens]
            out.append(ids)
    return out, cut


def pad(seqs, pad_id, device):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)


@torch.no_grad()
def embed_many(model, seqs, side, pad_id, device, batch=256):
    """Length-sorted batches for speed; returned in the input order."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = torch.empty((len(seqs), SETTINGS["projection_dim"]), dtype=torch.float32, device=device)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        ids, mask = pad([seqs[i] for i in idx], pad_id, device)
        out[torch.tensor(idx, device=device)] = model.embed(ids, mask, side)
    return out


def ranks_of_true(text_vecs, mol_vecs, true_idx, chunk=512):
    """1 + the number of other candidates scoring equal or higher; and the tie count."""
    ranks, ties = [], []
    for s in range(0, text_vecs.size(0), chunk):
        sims = text_vecs[s:s + chunk] @ mol_vecs.t()
        ti = true_idx[s:s + chunk]
        true = sims.gather(1, ti.unsqueeze(1))
        higher = (sims > true).sum(1)
        equal = (sims == true).sum(1) - 1
        ranks.append(1 + higher + equal)
        ties.append(equal)
    return torch.cat(ranks).cpu().numpy(), torch.cat(ties).cpu().numpy()


def metrics(ranks):
    r = np.asarray(ranks, dtype=np.float64)
    return {"hits@1": float(np.mean(r <= 1)), "hits@10": float(np.mean(r <= 10)),
            "mrr": float(np.mean(1.0 / r)), "mean_rank": float(np.mean(r)), "n": int(r.size)}


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=("bert", "gpt2"), required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--expect-step", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop early (smoke tests only; the manifest records it)")
    a = ap.parse_args()

    dist.init_process_group("nccl", timeout=dt.timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world != SETTINGS["world_size"] and a.max_steps is None:
        raise SystemExit(f"world size {world}, SETTINGS fixes {SETTINGS['world_size']}")
    per_rank = SETTINGS["global_batch_pairs"] // world

    seed = SETTINGS["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.makedirs(a.out_dir, exist_ok=True)
    if rank == 0 and os.path.exists(os.path.join(a.out_dir, "metrics.json")):
        raise SystemExit(f"{a.out_dir} already holds a finished run")

    from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer
    tokenizer = MoleculeNatLangTokenizer().tokenizer
    pad_id = tokenizer.eos_token_id

    rows = [json.loads(line) for line in open(a.data, encoding="utf-8") if line.strip()]
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
    counts = {s: len(v) for s, v in by_split.items()}
    counts["candidates"] = len(rows)
    log(f"ChEBI-20 rows: {counts}")
    mt, dtpl, mx = SETTINGS["molecule_template"], SETTINGS["description_template"], \
        SETTINGS["max_tokens_per_side"]
    tok, truncated = {}, {}
    for s, items in by_split.items():
        tok[s + "_text"], truncated[s + "_text"] = encode_all(
            tokenizer, [dtpl.format(description=r["description"]) for r in items], mx)
    tok["all_mol"], truncated["all_mol"] = encode_all(
        tokenizer, [mt.format(smiles=r["smiles"]) for r in rows], mx)
    index_of = {r["cid"]: i for i, r in enumerate(rows)}
    split_mol = {s: [tok["all_mol"][index_of[r["cid"]]] for r in items]
                 for s, items in by_split.items()}
    lengths = {k: {"max": max(map(len, v)), "mean": float(np.mean([len(x) for x in v]))}
               for k, v in tok.items()}
    log(f"truncated at {mx} tokens: {truncated}")

    # ---- model
    encoder = BertEncoder(a.init, a.expect_step) if a.arch == "bert" \
        else GPT2Encoder(a.init, a.expect_step)
    if SETTINGS["gradient_checkpointing"]:
        if a.arch == "bert":
            encoder.bert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            encoder.checkpointing = True
    torch.manual_seed(seed)  # the two maps start from the same draw on every rank and arm
    model = DualEncoder(encoder).to(device)
    n_params = sum(p.numel() for p in model.encoder.parameters())
    log(f"encoder parameters: {n_params:,}; loaded {encoder.loaded}")
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local])

    def no_decay(name, p):
        return p.ndim < 2 or "LayerNorm" in name or "ln_" in name or name.endswith("logit_scale")

    groups = {("enc", True): [], ("enc", False): [], ("head", True): [], ("head", False): []}
    for name, p in model.named_parameters():
        part = "enc" if name.startswith("encoder.") else "head"
        groups[(part, not no_decay(name, p))].append(p)
    lr_of = {"enc": SETTINGS["lr_encoder"], "head": SETTINGS["lr_head"]}
    opt = torch.optim.AdamW(
        [{"params": ps, "lr": lr_of[part], "weight_decay": SETTINGS["weight_decay"] if dec else 0.0}
         for (part, dec), ps in groups.items() if ps],
        betas=(0.9, 0.999), eps=1e-8)

    n_train = counts["train"]
    steps_per_epoch = n_train // SETTINGS["global_batch_pairs"]
    total = steps_per_epoch * SETTINGS["epochs"]
    warmup = int(SETTINGS["warmup_fraction"] * total)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    log(f"steps/epoch {steps_per_epoch}, total {total}, warmup {warmup}")

    # ---- evaluation on valid
    def evaluate():
        model.eval()
        with torch.no_grad():
            tv = embed_many(model, tok["valid_text"], "text", pad_id, device)
            gv = embed_many(model, split_mol["valid"], "mol", pad_id, device)
            scale = model.logit_scale.exp().clamp(max=SETTINGS["logit_scale_max"])
            b = SETTINGS["eval_block_pairs"]
            tot, n = 0.0, 0
            for s in range(0, tv.size(0), b):
                k = min(b, tv.size(0) - s)
                tot += clip_loss(tv[s:s + k], gv[s:s + k], scale).item() * k
                n += k
            rk, _ = ranks_of_true(tv, gv, torch.arange(tv.size(0), device=device))
        model.train()
        return tot / n, metrics(rk)["mrr"], float(scale)

    curve = []
    best = {"loss": float("inf"), "step": None}
    best_path = os.path.join(a.out_dir, "best.pt")

    def record_eval(step):
        vloss, vmrr, scale = evaluate()
        curve.append({"step": step, "eval_loss": vloss, "eval_mrr_valid3301": vmrr,
                      "logit_scale": scale})
        log(f"step {step}: valid loss {vloss:.6f}  valid MRR(3,301) {vmrr:.4f}  scale {scale:.2f}")
        if vloss < best["loss"]:
            best.update(loss=vloss, step=step)
            if rank == 0:
                torch.save({"model": model.state_dict(), "step": step}, best_path + ".tmp")
                os.replace(best_path + ".tmp", best_path)
        dist.barrier()

    # ---- train
    model.train()
    record_eval(0)
    g = torch.Generator()
    step, t0 = 0, time.time()
    train_log = []
    stop = a.max_steps if a.max_steps is not None else total
    for epoch in range(SETTINGS["epochs"]):
        g.manual_seed(seed + epoch)
        perm = torch.randperm(n_train, generator=g).tolist()
        for b in range(steps_per_epoch):
            if step >= stop:
                break
            idx = perm[b * SETTINGS["global_batch_pairs"]:(b + 1) * SETTINGS["global_batch_pairs"]]
            mine = idx[rank * per_rank:(rank + 1) * per_rank]
            seqs = [tok["train_text"][i] for i in mine] + [split_mol["train"][i] for i in mine]
            ids, mask = pad(seqs, pad_id, device)
            t_loc, g_loc, scale = ddp(ids, mask, len(mine))
            t_all = torch.cat(dnn.all_gather(t_loc), 0)
            g_all = torch.cat(dnn.all_gather(g_loc), 0)
            loss = clip_loss(t_all, g_all, scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            train_log.append({"step": step, "epoch": epoch + (b + 1) / steps_per_epoch,
                              "train_loss": loss.item(), "grad_norm": float(gn),
                              "lr_encoder": sched.get_last_lr()[0]})
            if step % 10 == 0:
                log(f"step {step}/{total} epoch {epoch} loss {loss.item():.4f} "
                    f"gn {float(gn):.3f} {(time.time() - t0) / step:.2f}s/step")
            if step % SETTINGS["eval_every_steps"] == 0 or step == total:
                record_eval(step)
        if step >= stop:
            break
    if step % SETTINGS["eval_every_steps"] != 0 and step != total:
        record_eval(step)

    # ---- test ranking with the selected checkpoint
    dist.barrier()
    sd = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()
    out = {}
    if rank == 0:
        with torch.no_grad():
            mol_vecs = embed_many(model, tok["all_mol"], "mol", pad_id, device)
            text_vecs = embed_many(model, tok["test_text"], "text", pad_id, device)
            true_idx = torch.tensor([index_of[r["cid"]] for r in by_split["test"]], device=device)
            rk, ties = ranks_of_true(text_vecs, mol_vecs, true_idx)
        out = metrics(rk)
        with open(os.path.join(a.out_dir, "ranks.tsv"), "w", encoding="utf-8") as fh:
            fh.write("description_id\tarm\tcorrect_molecule_cid\tcorrect_molecule_smiles\t"
                     "rank\tties\tn_candidates\n")
            for r, k, t in zip(by_split["test"], rk, ties):
                fh.write(f"{r['cid']}\t{a.arm}\t{r['cid']}\t{r['smiles']}\t{int(k)}\t{int(t)}\t"
                         f"{len(rows)}\n")
        tl = {r["step"]: r for r in train_log}
        with open(os.path.join(a.out_dir, "finetune_curve.tsv"), "w", encoding="utf-8") as fh:
            fh.write("arm\tstep\tepoch\ttrain_loss\teval_loss\teval_mrr_valid3301\tlogit_scale\t"
                     "grad_norm\tlr_encoder\n")
            ev = {c["step"]: c for c in curve}
            for s in sorted(set(tl) | set(ev)):
                t, e = tl.get(s, {}), ev.get(s, {})
                fh.write("\t".join([
                    a.arm, str(s), f"{t['epoch']:.4f}" if t else "0",
                    f"{t['train_loss']:.6f}" if t else "",
                    f"{e['eval_loss']:.6f}" if e else "",
                    f"{e['eval_mrr_valid3301']:.6f}" if e else "",
                    f"{e['logit_scale']:.4f}" if e else "",
                    f"{t['grad_norm']:.4f}" if t else "",
                    f"{t['lr_encoder']:.3e}" if t else ""]) + "\n")
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True, check=True).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                        capture_output=True, text=True).stdout.strip())
        except Exception:
            commit, dirty = None, None
        manifest = {
            "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "arm": a.arm, "arch": a.arch, "init": a.init, "expect_step": a.expect_step,
            "loaded": encoder.loaded, "encoder_parameters": n_params,
            "dropout_from_pretraining": encoder.dropout,
            "job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
            "git": {"commit": commit, "dirty": dirty}, "world_size": world,
            "per_rank_pairs": per_rank, "settings": SETTINGS, "counts": counts,
            "truncated": truncated, "token_lengths": lengths,
            "steps_per_epoch": steps_per_epoch, "total_steps": total, "warmup_steps": warmup,
            "steps_run": step, "max_steps_override": a.max_steps,
            "selected_step": best["step"], "selected_valid_loss": best["loss"],
            "test": out, "data": os.path.abspath(a.data),
        }
        with open(os.path.join(a.out_dir, "run_manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        log(f"TEST {a.arm} selected step {best['step']}: {out}")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
