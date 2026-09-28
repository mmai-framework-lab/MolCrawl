"""Reduce the BERT grid runs to one row each: the numbers a result is made of.

A grid of loss curves is raw data. What a report needs is each arm as a point --
did it hold or collapse, where, how far down it got, and what it reads at a fixed
amount of data -- so that the relationship between learning rate and model size
can be drawn instead of the curves themselves.

Paths live in the config file, not here: this repository is public.

    python scripts/bert_grid_summary.py --config <file.json> --out <file.tsv>

The config names the run roots and, per modality, the model-free floor (the loss
of a model that learned nothing). Sizes come from each checkpoint's own config,
and the parameter count is summed from the safetensors header, so it is the model
that actually trained rather than a number recomputed from the shape.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

TOKENS_PER_STEP = 2_560 * 1_024  # 2,560 sequences/step, sequence length 1,024
PF_DAY = 8.64e19  # 1 petaflop/s-day = 1e15 operations/s x 86,400 s


def non_embedding(params, conf):
    """Parameters that enter the matrix multiplies.

    The 6ND convention counts these, not the embeddings: a large vocabulary adds
    parameters that barely add work. molnl's 50,264 tokens put 39M into embeddings,
    which would otherwise make its small model look 1.45x protein's while costing
    the same to train.
    """
    h = conf.get("hidden_size")
    v = conf.get("vocab_size")
    pos = conf.get("max_position_embeddings", 0)
    if not (params and h and v):
        return None
    return params - (v * h + pos * h + 2 * h)


def newest_checkpoint(run_dir: Path) -> Path | None:
    cks = [p for p in run_dir.glob("checkpoint-*") if (p / "trainer_state.json").exists()]
    return max(cks, key=lambda p: int(p.name.split("-")[1])) if cks else None


def param_count(ckpt: Path) -> int | None:
    """Sum the tensor shapes in the safetensors header. No weights are read."""
    path = ckpt / "model.safetensors"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        n = 1
        for dim in meta["shape"]:
            n *= dim
        total += n
    return total


def series(ckpt: Path, key: str = "eval_loss_mask"):
    st = json.loads((ckpt / "trainer_state.json").read_text())
    pts = sorted((e["step"], e[key]) for e in st["log_history"] if key in e)
    return [p[0] for p in pts], [p[1] for p in pts]


def at_tokens(steps, values, budget_tokens):
    """The value at the last evaluation that fits inside the budget."""
    keep = [(s, v) for s, v in zip(steps, values) if s * TOKENS_PER_STEP <= budget_tokens]
    return keep[-1][1] if keep else None


def verdict(steps, values, floor):
    """(collapsed, step it turned, best before that, best overall).

    Two ways an arm ends up unusable, and both have to be caught: it can come back
    to the model-free floor, or it can end far above its own best without reaching
    the floor. The step it turned is the first evaluation after the best where it
    crosses back.
    """
    best_i = min(range(len(values)), key=lambda i: values[i])
    best, last = values[best_i], values[-1]
    back_at_floor = floor is not None and last >= floor - 0.005
    far_above_best = last > 1.5 * best
    if not (back_at_floor or far_above_best):
        return False, None, best, best
    limit = (floor - 0.005) if back_at_floor else 1.5 * best
    turned = next((steps[i] for i in range(best_i, len(values)) if values[i] >= limit), None)
    return True, turned, best, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    budgets = cfg.get("budgets_tokens", [])

    rows = []
    for group in cfg["groups"]:
        floor = group.get("floor")
        for run_dir in sorted(Path(group["root"]).glob(group["glob"])):
            m = re.search(group["derive"], run_dir.name)
            if not m:
                continue
            ckpt = newest_checkpoint(run_dir)
            if ckpt is None:
                print(f"  ! no checkpoint: {run_dir.name}", file=sys.stderr)
                continue
            steps, values = series(ckpt)
            if not steps:
                continue
            collapsed, turned, best_before, best = verdict(steps, values, floor)
            conf = json.loads((ckpt / "config.json").read_text())
            row = {
                "modality": group["modality"],
                "run": run_dir.name,
                "size": m.group("size"),
                "lr": float(m.group("lr").replace("p", ".").replace("e", "e-")),
                "params": param_count(ckpt),
                "params_nonembed": non_embedding(param_count(ckpt), conf),
                "hidden": conf.get("hidden_size"),
                "layers": conf.get("num_hidden_layers"),
                "vocab": conf.get("vocab_size"),
                "last_step": steps[-1],
                "last_tokens": steps[-1] * TOKENS_PER_STEP,
                "last_pfdays": round(6 * (non_embedding(param_count(ckpt), conf) or 0)
                                     * steps[-1] * TOKENS_PER_STEP / PF_DAY, 3),
                "last_loss": f"{values[-1]:.4f}",
                "best_loss": f"{best:.4f}",
                "best_step": steps[values.index(best)],
                "collapsed": int(collapsed),
                "turned_step": turned or "",
                "turned_tokens": (turned * TOKENS_PER_STEP) if turned else "",
                "floor": floor if floor is not None else "",
            }
            for b in budgets:
                got = at_tokens(steps, values, b)
                row[f"loss@{b / 1e9:g}G"] = f"{got:.4f}" if got is not None else ""
            rows.append(row)
            print(f"    {group['modality']:9s} {row['size']:6s} lr={row['lr']:<9g} "
                  f"N={row['params']:>11,} last={values[-1]:.4f} best={best:.4f} "
                  f"{'崩壊' if collapsed else '健全'}", file=sys.stderr)

    columns = list(rows[0]) if rows else []
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")
    print(f"wrote {args.out} ({len(rows)} arms)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
