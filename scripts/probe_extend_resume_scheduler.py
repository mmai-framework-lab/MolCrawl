"""What a BERT extension's resume actually restores, measured rather than read.

An extension raises ``max_steps`` and switches the schedule to
``constant_with_warmup`` at a fraction of the original peak. HF restores the
scheduler from the checkpoint, and ``LambdaLR.load_state_dict`` does
``self.__dict__.update(state_dict)`` -- which brings back ``base_lrs`` and
``last_epoch``, not just a step counter. So the extension's own
``learning_rate`` and ``warmup_steps`` can be silently discarded.

``workflows/bert-grid.sbatch`` answers that by refusing to resume while
``scheduler.pt`` is present, and tells the caller to remove it from the copy,
"keep optimizer.pt". But in transformers 4.45.1 one ``if`` guards both loads::

    if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
        ...
        self.optimizer.load_state_dict(...)
        self.lr_scheduler.load_state_dict(...)

so removing the file may also skip the optimizer -- the Adam moments a
continuation needs. This probe settles it by running all three ways of handing
over a checkpoint and reporting, for each, what arrived:

  keep     scheduler.pt as the previous schedule saved it
  remove   scheduler.pt deleted, which is what bert-grid.sbatch instructs
  rewrite  scheduler.pt kept but re-pointed at the extension's own rate

A fresh Adam has an empty ``state``, and it is only populated by a load or by
the first step, so ``len(optimizer.state)`` read before training starts is a
direct answer to "were the moments restored".

Tiny model, random tokens, CPU. Nothing here depends on the rna data.
"""

import argparse
import json
import os
import shutil
import tempfile

import torch
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from molcrawl.core.torch_compat import enable_full_torch_load

# Every real entry point calls this at import time. transformers 4.45.1 reloads
# rng_state.pth without passing weights_only, so under torch >= 2.6 a resume dies
# on the numpy RNG state unless the pre-2.6 default is restored. A probe that
# skipped it would not be testing the resume the production runs take.
enable_full_torch_load()

VOCAB = 64
SEQ = 16


class ToyTokens(Dataset):
    """Random ids with labels equal to them: a loss to descend, nothing more."""

    def __init__(self, rows, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.ids = torch.randint(0, VOCAB, (rows, SEQ), generator=gen)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ids = self.ids[i]
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }


class RecordAtTrainBegin(TrainerCallback):
    """Read the optimizer and the scheduler before the first step runs."""

    def __init__(self):
        self.seen = {}

    def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kw):
        moments = 0.0
        for st in getattr(optimizer, "state", {}).values():
            if "exp_avg" in st:
                moments += float(st["exp_avg"].abs().sum())
        self.seen = {
            "optimizer_state_entries": len(getattr(optimizer, "state", {})),
            "exp_avg_abs_sum": moments,
            "base_lrs": [float(x) for x in getattr(lr_scheduler, "base_lrs", [])],
            "last_epoch": getattr(lr_scheduler, "last_epoch", None),
            "global_step": state.global_step,
        }
        return control


def build_model():
    cfg = BertConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=SEQ,
    )
    return BertForMaskedLM(cfg)


def args_for(out_dir, max_steps, lr, warmup, sched, save_steps):
    kw = dict(
        output_dir=out_dir,
        max_steps=max_steps,
        learning_rate=lr,
        warmup_steps=warmup,
        per_device_train_batch_size=4,
        save_steps=save_steps,
        save_strategy="steps",
        logging_steps=1,
        report_to=[],
        seed=42,
        disable_tqdm=True,
        save_safetensors=True,
    )
    if sched is not None:
        kw["lr_scheduler_type"] = sched
    # The flag is use_cpu in 4.45; no_cuda in older ones.
    try:
        return TrainingArguments(use_cpu=True, **kw)
    except TypeError:
        return TrainingArguments(no_cuda=True, **kw)


def rates(trainer, after=None):
    """Logged rates, optionally only the ones this run added.

    HF restores log_history from the checkpoint, so on a resume the head of the
    list belongs to the ORIGINAL run. Reading it without a floor reports the
    previous schedule's warmup back as the extension's -- which the first version
    of this probe did, and it made all three hand-overs look identical.
    """
    out = [
        (h["step"], h["learning_rate"])
        for h in trainer.state.log_history
        if "learning_rate" in h
    ]
    if after is not None:
        out = [(s, lr) for s, lr in out if s > after]
    return out


def prepare(ckpt_src, dest, how, new_lr):
    """Copy the checkpoint and hand scheduler.pt over in one of three ways."""
    shutil.copytree(ckpt_src, dest)
    sched = os.path.join(dest, "scheduler.pt")
    if how == "keep":
        return
    if how == "remove":
        os.remove(sched)
        return
    if how == "rewrite":
        state = torch.load(sched, map_location="cpu", weights_only=False)
        state["base_lrs"] = [new_lr] * len(state["base_lrs"])
        state["last_epoch"] = 0
        state["_step_count"] = 1
        state["_last_lr"] = [0.0] * len(state["base_lrs"])
        if "_is_initial" in state:
            state["_is_initial"] = True
        torch.save(state, sched)
        return
    raise ValueError(how)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-steps", type=int, default=20)
    ap.add_argument("--add-steps", type=int, default=10)
    ap.add_argument("--peak", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    tenth = a.peak / 10.0
    data = ToyTokens(200)
    report = {
        "peak": a.peak,
        "extension_lr": tenth,
        "pre_steps": a.pre_steps,
        "warmup": a.warmup,
        "cases": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        # The original run: HF's default linear schedule, decaying to 0 at pre_steps.
        pre_dir = os.path.join(tmp, "pre")
        pre = Trainer(
            model=build_model(),
            args=args_for(pre_dir, a.pre_steps, a.peak, 2, None, a.pre_steps),
            train_dataset=data,
        )
        pre.train()
        tail = rates(pre)[-3:]
        print(f"=== original run: last logged rates {tail}")
        report["original_tail_rates"] = tail

        ckpt = os.path.join(pre_dir, f"checkpoint-{a.pre_steps}")
        saved = torch.load(
            os.path.join(ckpt, "scheduler.pt"), map_location="cpu", weights_only=False
        )
        print(f"=== saved scheduler.pt: base_lrs {saved['base_lrs']} "
              f"last_epoch {saved['last_epoch']} _last_lr {saved['_last_lr']}")
        report["saved_scheduler"] = {
            "base_lrs": [float(x) for x in saved["base_lrs"]],
            "last_epoch": saved["last_epoch"],
            "_last_lr": [float(x) for x in saved["_last_lr"]],
        }

        for how in ("keep", "remove", "rewrite"):
            out = os.path.join(tmp, f"ext-{how}")
            os.makedirs(out)
            dest = os.path.join(out, f"checkpoint-{a.pre_steps}")
            prepare(ckpt, dest, how, tenth)

            watch = RecordAtTrainBegin()
            ext = Trainer(
                model=build_model(),
                args=args_for(
                    out,
                    a.pre_steps + a.add_steps,
                    tenth,
                    a.warmup,
                    "constant_with_warmup",
                    a.pre_steps + a.add_steps,
                ),
                train_dataset=data,
                callbacks=[watch],
            )
            ext.train(resume_from_checkpoint=dest)
            first = rates(ext, after=a.pre_steps)[: a.warmup + 3]
            case = {
                "at_train_begin": watch.seen,
                "first_rates": first,
                "moments_restored": watch.seen["optimizer_state_entries"] > 0,
                "runs_at_requested_rate": bool(
                    watch.seen["base_lrs"]
                    and abs(watch.seen["base_lrs"][0] - tenth) < tenth * 1e-6
                ),
                "warmup_happened": bool(first and first[0][1] < tenth * 0.9),
            }
            report["cases"][how] = case
            print(f"\n=== {how}")
            print(f"    optimizer state entries at train begin : "
                  f"{watch.seen['optimizer_state_entries']}  "
                  f"(exp_avg abs sum {watch.seen['exp_avg_abs_sum']:.6g})")
            print(f"    scheduler base_lrs / last_epoch        : "
                  f"{watch.seen['base_lrs']} / {watch.seen['last_epoch']}")
            print(f"    first logged rates                     : {first}")
            print(f"    moments restored                       : {case['moments_restored']}")
            print(f"    runs at the requested {tenth:g}            : "
                  f"{case['runs_at_requested_rate']}")
            print(f"    warmup actually ramped                 : {case['warmup_happened']}")

    print("\n=== verdict")
    for how, case in report["cases"].items():
        print(f"    {how:8s} moments={case['moments_restored']!s:5s} "
              f"rate_ok={case['runs_at_requested_rate']!s:5s} "
              f"warmup={case['warmup_happened']!s:5s}")

    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
