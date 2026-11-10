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

Two parameter columns, named apart because they have been mixed up.
``params_total`` is every parameter in the checkpoint. ``params_nonembedding``
is the 6ND count the figures put on their compute axis, and it is the smaller
one: a 616-token vocabulary adds parameters that barely add work. The two
conventions differ by architecture and that is deliberate -- nanoGPT ties
``lm_head`` to ``wte``, so the token embedding does enter a matrix multiply
there and is not subtracted, while BERT's is.

The caveat that goes with these four values (2026-11-09 judgement §1.1). Each
stage begins with a 500-step warmup and ``constant_with_warmup`` ramps it from
zero, so there is a stretch at the head of every stage where the rate is below
the stage's own constant. At step 15,000 that follows a stage which decayed to
zero, but at step 30,000 it follows a stage that ended at a tenth of the peak
and holding it -- the rate drops back to zero there and climbs over 500 steps
anyway, because the ramp was kept so that the 30,000-step values and the
45,000-step ones were measured under the same shape. Two of the four arms get a
worse validation point just after the 30,000 boundary and two do not, so nothing
is concluded from it; it is stated because the values are read as one schedule.

That drop is measured, not inferred: the rate Trainer logged at the first point
after each of the eight stage heads is a fifth of that stage's constant, which is
the 100th step of a 500-step ramp from zero. ``scripts/lr_at_stage_head_tsv.py``
reads it back. It is worth measuring because the 45,000-step configs describe it
the other way round -- "there is no jump from nothing at this resume point as
there was at 15,000" is about the rate the previous stage ENDED on, not about
where the new ramp begins.

**What the caveat applies to, and what it does not.** It applies to a difference read
ACROSS stages, and to nothing else (2026-11-10 judgement §1.1). The adopted values
themselves are not inside a ramp: the four adopted steps are 38,900 to 44,800, which
is 8,900 steps or more past the 30,000 head.

**A column of stage values is a column of stage minima.** Where these four values are
tabulated beside a 15,000 and a 30,000 column, every one of those columns holds that
stage's lowest point, not the value at the boundary step. On large the 30,000 column
is 0.065906 at step 29,800 while the boundary step itself is 0.066789. Anyone adding
the caveat to such a table has to keep that straight, or the note lands on a number
it is not about.

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

# (size, learning rate as written in the config, run directory, the stage before it)
#
# The fourth value is the run this one continued. The improvement a stage bought is
# worth a column of its own: for large it is -0.000078, smaller than the gaps between
# neighbouring sizes in this very table, and a value shown without it reads as though
# the longer schedule had bought something it did not. It is read from that run rather
# than written in, so a further stage needs no editing here.
BERT_ARMS = [
    ("small", "1e-4", "compounds-small-lr1e3-seed42-ext45k",
     "compounds-small-lr1e3-seed42-ext30k"),
    ("medium", "2.8e-5", "compounds-medium-lr2p8e4-seed42-ext45k",
     "compounds-medium-lr2p8e4-seed42-ext30k"),
    ("large", "1.5e-5", "compounds-large-lr1p5e4-seed42-ext45k",
     "compounds-large-lr1p5e4-seed42-ext30k"),
    ("xlarge", "4.5e-6", "compounds-xl-lr4p5e5-seed42-ext45k",
     "compounds-xl-lr4p5e5-seed42-ext30k"),
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
    "params_total",
    "params_nonembedding",
    "hidden",
    "layers",
    "learning_rate",
    "steps_run",
    "lr_at_last_step",
    "lr_ends_at_zero",
    "metric",
    "adopted_value",
    "adopted_step",
    "adopted_epochs",
    "adopted_tokens",
    "best_is_final_eval",
    "final_minus_best",
    "prev_stage_value",
    "prev_stage_steps",
    "minus_prev_stage",
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


def bert_nonembedding(params: int, config: dict) -> int:
    """The 6ND count: the token and position embeddings and the embedding layer norm
    are taken out, matching scripts/bert_grid_summary.non_embedding."""
    h = config["hidden_size"]
    return params - (config["vocab_size"] * h + config["max_position_embeddings"] * h + 2 * h)


def gpt2_nonembedding(params: int, config: dict) -> int:
    """The 6ND count for nanoGPT: only the position embedding comes out.

    ``wte`` is tied to ``lm_head``, so those parameters sit in the output matrix
    multiply and subtracting them would understate the compute.
    """
    return params - config["n_positions"] * config["n_embd"]


def bert_final_rate(state: dict) -> tuple[str, str]:
    """The learning rate the last logged step ran at, and whether it reached zero.

    The compounds schedule is three stages: 1 to 15,000 decays linearly to zero,
    and each later stage warms up over 500 steps and then holds a tenth of the
    stage-1 peak. log_history is restored on resume, so the zero at the end of
    stage 1 is still in it -- the question is what the *last* step ran at, which
    is why this reads the end of the list rather than its minimum.
    """
    rates = [h["learning_rate"] for h in state["log_history"] if "learning_rate" in h]
    if not rates:
        return "", ""
    return f"{rates[-1]:.3e}", "yes" if rates[-1] == 0 else "no"


def last_checkpoint(run: Path) -> Path:
    steps = sorted(
        int(p.name.split("-")[1]) for p in run.glob("checkpoint-*") if p.is_dir()
    )
    if not steps:
        raise SystemExit(f"no checkpoint-* directory under {run}")
    return run / f"checkpoint-{steps[-1]}"


def bert_best(run: Path) -> tuple[float, int, int]:
    """The smallest eval_loss_mask in a run, its step, and the run's last step.

    Read from log_history rather than from best_metric, so that a stage whose
    best_metric fell back to another metric cannot quietly supply the number.
    """
    checkpoint = last_checkpoint(run)
    with open(checkpoint / "trainer_state.json") as fh:
        state = json.load(fh)
    evals = [h for h in state["log_history"] if "eval_loss_mask" in h]
    if not evals:
        raise SystemExit(f"{run}: no eval_loss_mask in log_history")
    best = min(evals, key=lambda h: h["eval_loss_mask"])
    return best["eval_loss_mask"], int(best["step"]), int(state["global_step"])


def bert_row(root: Path, size: str, lr: str, dirname: str, prev: str) -> list[str]:
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
    prev_value, _, prev_steps = bert_best(root / prev)
    return [
        "bert",
        size,
        str(total := safetensors_param_count(checkpoint)),
        str(bert_nonembedding(total, config)),
        str(config["hidden_size"]),
        str(config["num_hidden_layers"]),
        lr,
        str(state["global_step"]),
        *bert_final_rate(state),
        "eval_loss_mask",
        value,
        str(step),
        f"{epochs(step):.2f}",
        str(tokens(step)),
        "yes" if final["step"] == scanned["step"] else "no",
        f"{final['eval_loss_mask'] - scanned['eval_loss_mask']:+.6f}",
        f"{prev_value:.6f}",
        str(prev_steps),
        f"{scanned['eval_loss_mask'] - prev_value:+.6f}",
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
        str(total := gpt2_param_count(config)),
        str(gpt2_nonembedding(total, config)),
        str(config["n_embd"]),
        str(config["n_layer"]),
        lr,
        str(run_steps),
        # nanoGPT's training_args.json records the configured base rate, not the
        # value the step ran at, so there is nothing here to report truthfully.
        "",
        "",
        "val_loss",
        f"{best_value:.6f}",
        str(best_step),
        f"{epochs(best_step):.2f}",
        str(tokens(best_step)),
        "yes" if best_step == run_steps else "no",
        f"{final_value - best_value:+.6f}",
        "",  # no stage before it: these runs were not continued from a shorter one
        "",
        "",
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
