#!/usr/bin/env python3
"""The table the 2026-11-20 directive's §9 asks for, built from the inventory.

§9: before submitting, publish every one of the 68 arms with one GPU's block count
and its accumulation count, and the measured step times wherever the shape changed.

Both halves are read rather than written out by hand: the arms come from the derived
inventory TSV, and the step times from the SHAPE_RESULT lines the §4 probe jobs print
into their own logs. A table typed from a screen is how a grid comes to be submitted
at a shape the table does not show.

    molnl_batch_sweep_shapes_table.py --arms rt1-...tsv \
        [--probe-logs workflows/slurm-logs/mc-molnl-bshape-*.out] \
        [--current-set current] [--out <path>.md]

The output is markdown, for the report the directive asks for; the numbers it holds
are the inventory's own cells.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import re
import sys

#: ``SHAPE_RESULT arch=bert label=medium-8x80 per_step_s=7.612 ... source=...``.
#: The value of the last key holds spaces, so the keys are found and the text between
#: them is the value.
KEY = re.compile(r"(\w+)=")

SIZES = ("small", "medium", "large", "xl")
BATCHES = (256, 512, 1024, 2048, 4096)


def todays_shapes():
    """One GPU's block count every molnl config carries today, from one place.

    scripts/molnl_batch_sweep_arms.py holds it as the 'current' set; reading it from
    there rather than restating it means the "today" column cannot drift away from
    the set the inventory was derived against.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "molnl_batch_sweep_arms.py")
    spec = importlib.util.spec_from_file_location("molnl_batch_sweep_arms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MICRO_BATCH["current"]


def read_arms(path):
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path}: no arm")
    return rows


def parse_shape_results(paths):
    """{(arch, size, micro): {field: value}} out of the probe logs."""
    out = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("SHAPE_RESULT "):
                    continue
                body = line[len("SHAPE_RESULT "):].rstrip("\n")
                keys = [(m.group(1), m.start(), m.end()) for m in KEY.finditer(body)]
                fields = {}
                for index, (name, _start, end) in enumerate(keys):
                    stop = keys[index + 1][1] if index + 1 < len(keys) else len(body)
                    fields[name] = body[end:stop].strip()
                label = fields.get("label", "")
                size, _, shape = label.partition("-")
                micro, _, accum = shape.partition("x")
                if not micro.isdigit():
                    continue
                # The accumulation count is in the label rather than in a field of its
                # own, so it has to be taken from there or the column comes out empty.
                out[(fields.get("arch", ""), size, int(micro))] = dict(
                    fields, accum=accum, probe_log=path)
    return out


def arms_table(rows):
    lines = ["| arch | size | batch | lr | x | one GPU | accumulation | GPUs | "
             "product | steps | eval | save |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ordered = sorted(rows, key=lambda r: (r["arch"], SIZES.index(r["size"]),
                                          int(r["batch"]), float(r["lr"])))
    for row in ordered:
        lines.append(
            f"| {row['arch']} | {row['size']} | {int(row['batch']):,} | {row['lr']} | "
            f"{float(row['lr_multiplier']):g} | {row['micro_batch']} | "
            f"{row['grad_accum']} | {row['world_size']} | "
            f"{int(row['expected_global_batch']):,} | {int(row['steps']):,} | "
            f"{row['eval_interval']} | {row['save_interval'] or '-'} |")
    return lines, ordered


def shapes_table(rows, measured, set_name):
    """One row per measured shape: the time, what the grid chose, what molnl runs now."""
    lines = ["| arch | size | one GPU | accumulation | s/step | median | stdev | "
             "steps read | chosen | molnl today |",
             "|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|"]
    chosen, todays = {}, todays_shapes()
    for row in rows:
        chosen[(row["arch"], row["size"])] = int(row["micro_batch"])
    for (arch, size, micro), fields in sorted(measured.items()):
        if not arch:
            continue
        mark = "yes" if chosen.get((arch, size)) == micro else ""
        now = "yes" if todays.get(arch, {}).get(size) == micro else ""
        failed = fields.get("failed", "")
        per_step = fields.get("per_step_s") or (f"failed (rc={failed})" if failed
                                                else "")
        lines.append(
            f"| {arch} | {size} | {micro} | {fields.get('accum', '')} | {per_step} | "
            f"{fields.get('median_s', '')} | {fields.get('stdev_s', '')} | "
            f"{fields.get('steps_read') or fields.get('iters_read', '')} | {mark} | "
            f"{now} |")
    if len(lines) == 2:
        lines.append("| (no SHAPE_RESULT line was given) | | | | | | | | | |")
    lines.append("")
    lines.append(f"The grid's block count per GPU comes from the '{set_name}' set of "
                 f"scripts/molnl_batch_sweep_arms.py; the last column is what every "
                 f"molnl config carries today.")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--probe-logs", nargs="*", default=[],
                    help="the §4 probe job logs holding the SHAPE_RESULT lines")
    ap.add_argument("--set-name", default="current",
                    help="the micro-batch set the inventory was derived with")
    ap.add_argument("--out", default=None, help="write the markdown here as well")
    args = ap.parse_args(argv)

    rows = read_arms(args.arms)
    paths = [p for pattern in args.probe_logs for p in sorted(glob.glob(pattern))]
    measured = parse_shape_results(paths)

    table, ordered = arms_table(rows)
    counts = {}
    for row in ordered:
        counts[row["arch"]] = counts.get(row["arch"], 0) + 1
    head = [f"## The {len(ordered)} arms, one GPU's blocks and the accumulation",
            "", *table, "",
            f"{len(ordered)} arms: " + ", ".join(f"{k} {v}" for k, v in
                                                 sorted(counts.items())) + ".",
            "",
            "## The shapes, as measured", "",
            *shapes_table(rows, measured, args.set_name)]
    # Every (arch, size, batch) cell must name one shape, or the table is not the
    # thing that was submitted.
    seen = {}
    for row in ordered:
        key = (row["arch"], row["size"])
        seen.setdefault(key, set()).add((row["micro_batch"], row["world_size"]))
    bad = {k: v for k, v in seen.items() if len(v) > 1}
    if bad:
        raise SystemExit(f"a size has more than one shape in the inventory: {bad}")
    missing = [key for key in seen
               if not any(k[0] == key[0] and k[1] == key[1] for k in measured)]
    text = "\n".join(head) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)
    if paths:
        print(f"\nprobe logs read: {len(paths)}")
        for key in sorted(missing):
            print(f"  ** no measured shape for {key[0]} {key[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
