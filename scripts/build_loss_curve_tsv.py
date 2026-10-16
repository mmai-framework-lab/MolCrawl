"""Collect training and validation loss into one table, one row per evaluation point.

A loss curve lives in a different place for each architecture, and neither place is a
table: nanoGPT writes a three-column csv per run, and the HF Trainer keeps its history
inside the checkpoint's trainer_state.json. This reads whichever is present and emits
one long-format row per point, with a `split` column saying whether the value is the
training or the validation loss, so train and validation can be plotted from the same
file without a second pass.

Long format rather than two value columns, because a point can exist on one split and
not the other: nanoGPT reports both at every eval, while the HF history logs the
training loss on its own cadence and the validation loss only when it evaluates.

The training loss is not comparable across sizes. Both paths estimate it on a sample
drawn afresh at each point, and nanoGPT's sample size is `eval_iters * batch_size`,
which moves with the model. The column is here for the shape of the curve.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def from_nanogpt_csv(path):
    """nanoGPT's logging_*.csv: iter, train_loss, val_loss."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue
            try:
                step = int(row[0].strip())
                tr = float(row[1].strip())
                va = float(row[2].strip())
            except ValueError:
                continue                        # the header
            out.append((step, "train", tr))
            out.append((step, "valid", va))
    return out


def from_trainer_state(path, val_metric):
    """The HF Trainer's log_history, which lives inside a checkpoint."""
    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    out = []
    for e in state.get("log_history", []):
        step = e.get("step")
        if step is None:
            continue
        if "loss" in e:
            out.append((int(step), "train", float(e["loss"])))
        if val_metric in e:
            out.append((int(step), "valid", float(e[val_metric])))
    return out


def newest_trainer_state(run_dir):
    """The trainer_state.json of the furthest checkpoint, which holds the whole history."""
    cks = [d for d in glob.glob(os.path.join(run_dir, "checkpoint-*")) if os.path.isdir(d)]
    if not cks:
        return None
    cks.sort(key=lambda d: int(os.path.basename(d).split("-")[1]))
    cand = os.path.join(cks[-1], "trainer_state.json")
    return cand if os.path.exists(cand) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, metavar="ARCH:SIZE:LR:SEED:DIR",
                    help="one run; repeat for several")
    ap.add_argument("--val-metric", default="eval_loss_mask",
                    help="which HF metric is the validation loss for this architecture")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = []
    for spec in args.run:
        arch, size, lr, seed, run_dir = spec.split(":", 4)
        points = []
        source = None
        csvs = sorted(glob.glob(os.path.join(run_dir, "logging_*.csv")))
        if csvs:
            points = from_nanogpt_csv(csvs[-1])
            source = os.path.basename(csvs[-1])
        else:
            ts = newest_trainer_state(run_dir)
            if ts:
                points = from_trainer_state(ts, args.val_metric)
                source = os.path.relpath(ts, run_dir)
        if not points:
            print(f"  {arch} {size} lr={lr}: 点が取れない（{run_dir}）")
            continue
        for step, split, value in points:
            rows.append({"arch": arch, "size": size, "lr": lr, "seed": seed,
                         "step": step, "split": split, "value": f"{value:.6f}",
                         "source": source})
        n_tr = sum(1 for _, s, _ in points if s == "train")
        n_va = sum(1 for _, s, _ in points if s == "valid")
        print(f"  {arch:5s} {size:7s} lr={lr:8s} seed={seed:5s} train {n_tr:4d} 点 / "
              f"valid {n_va:4d} 点  ({source})")

    rows.sort(key=lambda r: (r["arch"], r["size"], r["lr"], r["seed"], r["step"], r["split"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arch", "size", "lr", "seed", "step",
                                           "split", "value", "source"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows):,} 行 → {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
