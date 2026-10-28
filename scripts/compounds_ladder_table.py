"""Build the compounds ladder table from the run directories themselves.

The adopted value is never taken from a report. For BERT it is read from
``trainer_state.json`` and cross-checked against an independent scan of
``eval_loss_mask`` in ``log_history``, because ``save_on_improve`` has been
seen to fall back to ``eval_loss``; a run whose ``best_metric`` disagrees with
that scan is written out with a ``MISMATCH`` marker rather than a number. For
GPT-2 it is read from ``training_args.json`` in the retained checkpoints, whose
``best_val_loss`` is the running best at the time each one was written.

Epochs and tokens are derived here rather than copied, from the training window
count and the global batch, so that a longer schedule needs no hand arithmetic.

Usage:
    compounds_ladder_table.py --bert-root DIR --gpt2-root DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

# The packed compounds build: training windows of 1,024 tokens, no padding.
TRAIN_WINDOWS = 398_917
SEQ_PER_STEP = 2_560
TOKENS_PER_SEQ = 1_024

# (size, learning rate as written in the config, run directory)
BERT_ARMS = [
    ("small", "1e-4", "compounds-small-lr1e3-seed42-ext30k"),
    ("medium", "2.8e-5", "compounds-medium-lr2p8e4-seed42-ext30k"),
    ("large", "1.5e-5", "compounds-large-lr1p5e4-seed42-ext30k"),
    ("xlarge", "4.5e-6", "compounds-xl-lr4p5e5-seed42-ext30k"),
]

GPT2_ARMS = [
    ("small", "8e-4", "compounds-small-lr8e4-seed42"),
    ("medium", "8e-4", "compounds-medium-lr8e4-seed42"),
    ("large", "1.6e-3", "compounds-large-lr1p6e3"),
    ("ex-large", "1.6e-3", "compounds-ex-large-lr1p6e3"),
]

COLUMNS = [
    "arch",
    "size",
    "params",
    "hidden",
    "layers",
    "learning_rate",
    "steps_run",
    "metric",
    "adopted_value",
    "adopted_step",
    "adopted_epochs",
    "adopted_tokens",
    "best_is_final_eval",
    "final_minus_best",
    "source",
]


def epochs(steps: int) -> float:
    return steps * SEQ_PER_STEP / TRAIN_WINDOWS


def tokens(steps: int) -> int:
    return steps * SEQ_PER_STEP * TOKENS_PER_SEQ


def safetensors_param_count(checkpoint: Path) -> int:
    """Sum the tensor shapes in a checkpoint's safetensors headers.

    Only the JSON header of each shard is read, so no weights are loaded.
    """
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        with open(index) as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    total, seen = 0, set()
    for shard in shards:
        with open(checkpoint / shard, "rb") as fh:
            length = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(length))
        for name, meta in head.items():
            if name == "__metadata__" or name in seen:
                continue
            seen.add(name)
            count = 1
            for dim in meta["shape"]:
                count *= dim
            total += count
    return total


def gpt2_param_count(config: dict) -> int:
    """Parameters implied by the declared dimensions of a nanoGPT model.

    nanoGPT ties ``lm_head`` to ``wte`` and these runs set ``bias = False``,
    so each block holds 12 H**2 in its matrices and 2 H in its layer norms.
    """
    hidden = config["n_embd"]
    layers = config["n_layer"]
    return (
        config["vocab_size"] * hidden
        + config["n_positions"] * hidden
        + layers * (12 * hidden * hidden + 2 * hidden)
        + hidden
    )


def last_checkpoint(run: Path) -> Path:
    steps = sorted(
        int(p.name.split("-")[1]) for p in run.glob("checkpoint-*") if p.is_dir()
    )
    if not steps:
        raise SystemExit(f"no checkpoint-* directory under {run}")
    return run / f"checkpoint-{steps[-1]}"


def bert_row(root: Path, size: str, lr: str, dirname: str) -> list[str]:
    run = root / dirname
    checkpoint = last_checkpoint(run)
    with open(checkpoint / "trainer_state.json") as fh:
        state = json.load(fh)
    with open(checkpoint / "config.json") as fh:
        config = json.load(fh)

    evals = [h for h in state["log_history"] if "eval_loss_mask" in h]
    if not evals:
        raise SystemExit(f"{run}: no eval_loss_mask in log_history")
    scanned = min(evals, key=lambda h: h["eval_loss_mask"])

    recorded = state.get("best_metric")
    agrees = recorded is not None and abs(recorded - scanned["eval_loss_mask"]) < 1e-12
    value = f"{scanned['eval_loss_mask']:.6f}" if agrees else "MISMATCH"

    step = int(scanned["step"])
    final = evals[-1]
    return [
        "bert",
        size,
        str(safetensors_param_count(checkpoint)),
        str(config["hidden_size"]),
        str(config["num_hidden_layers"]),
        lr,
        str(state["global_step"]),
        "eval_loss_mask",
        value,
        str(step),
        f"{epochs(step):.2f}",
        str(tokens(step)),
        "yes" if final["step"] == scanned["step"] else "no",
        f"{final['eval_loss_mask'] - scanned['eval_loss_mask']:+.6f}",
        f"{dirname}/{checkpoint.name}/trainer_state.json",
    ]


def gpt2_row(root: Path, size: str, lr: str, dirname: str) -> list[str]:
    run = root / dirname
    best_value, best_step, run_steps, final_value = None, None, 0, None
    for path in sorted(run.glob("checkpoint-*/training_args.json")):
        with open(path) as fh:
            args = json.load(fh)
        step = int(args["iteration"])
        if step >= run_steps:
            run_steps, final_value = step, args["val_loss"]
        if args["val_loss"] is not None and (
            best_value is None or args["val_loss"] < best_value
        ):
            best_value, best_step = args["val_loss"], step

    if best_value is None:
        raise SystemExit(f"{run}: no val_loss in any retained checkpoint")

    with open(last_checkpoint(run) / "config.json") as fh:
        config = json.load(fh)

    return [
        "gpt2",
        size,
        str(gpt2_param_count(config)),
        str(config["n_embd"]),
        str(config["n_layer"]),
        lr,
        str(run_steps),
        "val_loss",
        f"{best_value:.6f}",
        str(best_step),
        f"{epochs(best_step):.2f}",
        str(tokens(best_step)),
        "yes" if best_step == run_steps else "no",
        f"{final_value - best_value:+.6f}",
        f"{dirname}/checkpoint-*/training_args.json",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert-root", type=Path, required=True)
    parser.add_argument("--gpt2-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    opts = parser.parse_args()

    rows = [bert_row(opts.bert_root, *arm) for arm in BERT_ARMS]
    rows += [gpt2_row(opts.gpt2_root, *arm) for arm in GPT2_ARMS]

    # BOM so that the file is not taken for binary when it is shared on Slack.
    with open(opts.out, "w", encoding="utf-8-sig") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")

    for row in rows:
        print("\t".join(row))


if __name__ == "__main__":
    main()
