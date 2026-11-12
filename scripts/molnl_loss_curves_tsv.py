#!/usr/bin/env python3
"""The molnl loss curves of the adopted runs, one measured value per row.

The 2026-11-12 order asks for the training-side and validation-side loss of the runs this
workstream reports, BERT and GPT-2 in one file, at the interval each run recorded.

Two of its instructions decide the shape, and they are the exception the column
convention allows rather than its normal form:

  form A      ``value`` with a ``metric`` column is permitted only where the metric
              differs from row to row. It does here: BERT's validation side is
              eval_loss_mask over the 20% of positions it masked, GPT-2's is val_loss
              over every token it predicted, and the training sides are their
              counterparts. Four different numbers, which is why one named column could
              not hold them.
  split       train and valid stand on two rows at the same step, as the order asks,
              rather than in two columns.

Only the adopted runs go in -- one per size per architecture, the seven the retrieval
scoring was run on. molnl BERT xl is not among them: four of its rates held to 24,000
steps and none was adopted, so there is no one run to put in this file for that cell.
That absence is in the provenance rather than left to be noticed.

The values are read from what each run wrote: an HF run's trainer_state.json at its last
checkpoint, a nanoGPT run's logging_*.csv. ``--cross-check`` compares every row against
report/molnl-all-curves-train-and-val.tsv, the table that walks every run, so that this
file cannot disagree with the one already shipped.

    python scripts/molnl_loss_curves_tsv.py --bert-root <dir> --gpt2-root <dir> \
        --bert-runs <name>... --gpt2-runs <name>... --sizes <json> --out <tsv>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime

COLUMNS = ("arch", "size", "lr", "seed", "step", "split", "value", "metric", "source")
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "xl": 3}
ARCH_ORDER = {"bert": 0, "gpt2": 1}
SPLIT_ORDER = {"train": 0, "valid": 1}


def manifest(run_dir):
    path = os.path.join(run_dir, "run_manifest.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def lr_and_seed(run_dir):
    """The rate and seed the run recorded, not the ones its directory name suggests."""
    info = manifest(run_dir)
    lr = info.get("schedule", {}).get("learning_rate")
    seed = info.get("seed")
    if isinstance(seed, dict):
        seed = seed.get("seed")
    return (f"{lr:g}" if isinstance(lr, (int, float)) else ""), ("" if seed is None else seed)


def bert_rows(root, name, size):
    run = os.path.join(root, name)
    cks = sorted(glob.glob(os.path.join(run, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        raise SystemExit(f"{run}: no checkpoint to read a trainer_state.json from")
    state = json.load(open(os.path.join(cks[-1], "trainer_state.json"), encoding="utf-8"))
    source = f"{name}/{os.path.basename(cks[-1])}/trainer_state.json"
    lr, seed = lr_and_seed(run)
    rows = []
    for entry in state["log_history"]:
        step = entry.get("step")
        # The training loss is logged on its own steps, not on the evaluation ones.
        if "loss" in entry:
            rows.append(("bert", size, lr, seed, step, "train",
                         f"{entry['loss']:.6f}", "train_loss_mask", source))
        if "eval_loss_mask" in entry:
            rows.append(("bert", size, lr, seed, step, "valid",
                         f"{entry['eval_loss_mask']:.6f}", "eval_loss_mask", source))
    return rows, os.path.basename(cks[-1])


def gpt2_rows(root, name, size):
    run = os.path.join(root, name)
    paths = sorted(glob.glob(os.path.join(run, "logging_*.csv")))
    if not paths:
        raise SystemExit(f"{run}: no logging_*.csv to read the curve from")
    lr, seed = lr_and_seed(run)
    rows = []
    for path in paths:
        source = f"{name}/{os.path.basename(path)}"
        for row in csv.reader(open(path, encoding="utf-8")):
            if not row or not row[0].strip().isdigit():
                continue
            step = int(row[0])
            rows.append(("gpt2", size, lr, seed, step, "train",
                         f"{float(row[1]):.6f}", "train_loss", source))
            rows.append(("gpt2", size, lr, seed, step, "valid",
                         f"{float(row[2]):.6f}", "val_loss", source))
    return rows, "ckpt.pt"


def cross_check(rows, path, runs_by_frame):
    """Every row must match report/molnl-all-curves-train-and-val.tsv, row for row."""
    if not path or not os.path.exists(path):
        return ["the cross-check table was not given or does not exist"]
    want = {}
    for row in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        want[(row["run"], row["metric"], int(row["step"]))] = row["value"]
    problems = []
    for arch, size, _lr, _seed, step, _split, value, metric, _source in rows:
        run = runs_by_frame[(arch, size)]
        key = (run, metric, step)
        if key not in want:
            problems.append(f"{run} {metric} step {step}: not in the shipped table")
        elif abs(float(want[key]) - float(value)) > 5e-7:
            problems.append(f"{run} {metric} step {step}: {value} here, "
                            f"{want[key]} in the shipped table")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bert-root", required=True)
    ap.add_argument("--gpt2-root", required=True)
    ap.add_argument("--bert-runs", nargs="+", required=True,
                    help="<size>=<run directory name>, one per adopted size")
    ap.add_argument("--gpt2-runs", nargs="+", required=True)
    ap.add_argument("--sizes", required=True,
                    help="the json scripts/molnl_model_sizes.py wrote for these runs")
    ap.add_argument("--cross-check", default=None)
    ap.add_argument("--job", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def pairs(items):
        out = []
        for item in items:
            size, _, name = item.partition("=")
            if not name:
                raise SystemExit(f"--*-runs takes <size>=<run>, got {item!r}")
            out.append((size, name))
        return out

    rows, adopted, runs_by_frame = [], {}, {}
    for size, name in pairs(a.bert_runs):
        got, ckpt = bert_rows(a.bert_root, name, size)
        rows.extend(got)
        adopted[f"bert {size}"] = f"{name}/{ckpt}"
        runs_by_frame[("bert", size)] = name
        print(f"  bert {size:<7}{name:<28}{len(got):>7,} rows, adopted {ckpt}", flush=True)
    for size, name in pairs(a.gpt2_runs):
        got, ckpt = gpt2_rows(a.gpt2_root, name, size)
        rows.extend(got)
        adopted[f"gpt2 {size}"] = f"{name}/{ckpt}"
        runs_by_frame[("gpt2", size)] = name
        print(f"  gpt2 {size:<7}{name:<28}{len(got):>7,} rows, adopted {ckpt}", flush=True)

    problems = cross_check(rows, a.cross_check, runs_by_frame)
    if problems:
        raise SystemExit("the curves disagree with the shipped table: "
                         + "; ".join(problems[:10])
                         + (f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""))
    print(f"  {len(rows):,} rows, every one matching "
          f"{os.path.basename(a.cross_check or '')}", flush=True)

    sizes = json.load(open(a.sizes, encoding="utf-8"))
    rows.sort(key=lambda r: (ARCH_ORDER.get(r[0], 9), SIZE_ORDER.get(r[1], 9), r[4],
                             SPLIT_ORDER.get(r[5], 9)))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (columns: arch,size)\n")
        fh.write("# run: molecule_nat_lang, the adopted run of each size\n")
        for frame, path in adopted.items():
            fh.write(f"# checkpoint: {frame}: {path}\n")
        for name, entry in sorted(sizes.items(),
                                  key=lambda kv: (ARCH_ORDER.get(kv[1]["arch"], 9), kv[0])):
            line = (f"# params: {entry['arch']} {name}: non-embedding "
                    f"{entry['non_embedding']:,} of {entry['total']:,} total")
            if entry.get("non_embedding_compounds_rule"):
                line += (f"; by the compounds subtraction "
                         f"{entry['non_embedding_compounds_rule']:,}")
            fh.write(line + "\n")
        for name, entry in sorted(sizes.items()):
            if entry.get("tokens_per_step"):
                fh.write(f"# tokens_per_step: {entry['arch']} {name}: "
                         f"{entry['tokens_per_step']:,} = "
                         f"{entry['global_batch_sequences']:,} sequences x "
                         f"{entry['sequence_length']:,} tokens\n")
        fh.write(f"# source: {os.path.abspath(a.sizes)}\n")
        fh.write("# source: the source column names the file each row was read from, "
                 "relative to its run directory\n")
        if a.cross_check:
            fh.write(f"# source: cross-checked against {os.path.abspath(a.cross_check)}\n")
        if a.job:
            fh.write(f"# job: {a.job}\n")
        fh.write(f"# generated: "
                 f"{datetime.now().astimezone().isoformat(timespec='seconds')}"
                 f" by scripts/molnl_loss_curves_tsv.py\n")
        fh.write("# note: metric names what the row's value is. eval_loss_mask scores the "
                 "20% of positions BERT masked; val_loss scores every token GPT-2 "
                 "predicted; train_loss_mask and train_loss are their training-batch "
                 "counterparts. The four are not comparable with each other\n")
        fh.write("# note: BERT's training mask is drawn afresh every step; its evaluation "
                 "mask is fixed at seed 42. nanoGPT resamples its validation batches\n")
        fh.write("# note: molnl BERT xl has no row. Four of its rates held to 24,000 steps "
                 "and none was adopted, so the cell has no single run\n")
        fh.write("# note: the two non-embedding counts differ for GPT-2 only. This "
                 "repository's molnl figures use the first (wte and wpe both subtracted); "
                 "the compounds tables subtract wpe alone because wte is tied to lm_head\n")
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"  wrote {a.out}: {len(rows):,} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
