#!/usr/bin/env python3
"""Pick one GPU's block count per size out of the §4 probes, and record why.

§4 of the 2026-11-20 directive asks for the fastest value that can still build the
grid's smallest batch, one per size, held across the five batches -- measured, not
assumed, and reported with the measured numbers beside the chosen one.

This reads the SHAPE_RESULT lines the probe jobs printed and writes the choice as a
JSON file that scripts/molnl_batch_sweep_arms.py cross-checks its 'measured-1120' set
against at import. That set is a literal there -- loading it from here made the whole
derivation work only on the node that had run the probes -- and this file is its
provenance: it carries every measurement, which one won, by how much, and which job and
log it came from, and a disagreement with the literal stops the derivation.

Which number decides it:

  bert  per_step_s, HF's own train_runtime less the evaluation time inside it, over
        the probe's steps. One number per run, so there is nothing to average.
  gpt2  median_s over the per-iteration times past the warmup. nanoGPT logs every
        iteration, and the first measured probe had mean 3.472 against median 2.634
        with a standard deviation of 2.846: a few iterations carry something that is
        not the step, and the mean follows them.

    molnl_batch_shape_choice.py --probe-logs 'workflows/slurm-logs/mc-molnl-bshape-*.out' \
        --out scripts/molnl_batch_shape_choice_20261120.json

Read-only over the logs. It launches nothing and writes one file.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re

KEY = re.compile(r"(\w+)=")
JST = _dt.timezone(_dt.timedelta(hours=9))

SIZES = ("small", "medium", "large", "xl")
ARCHES = ("bert", "gpt2")

#: Which field ranks the shapes, per architecture, and why. The reason travels with
#: the number into the recorded file.
RANK_BY = {
    "bert": ("per_step_s",
             "HF train_runtime less the evaluation time inside it, over the steps"),
    "gpt2": ("median_s",
             "median of the per-iteration times past the warmup; the mean follows a "
             "few iterations that are not the step"),
}

#: §4: one GPU's blocks times the GPU count must divide 256, the grid's smallest batch.
CANDIDATES = (4, 8, 16, 32, 64)
GPUS = 4


def parse(paths):
    """[{arch, size, micro, accum, fields, log}] out of the probe job logs."""
    out = []
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
                size, _, shape = fields.get("label", "").partition("-")
                micro, _, accum = shape.partition("x")
                if not micro.isdigit() or size not in SIZES:
                    continue
                out.append({"arch": fields.get("arch", ""), "size": size,
                            "micro": int(micro), "accum": accum,
                            "fields": fields, "log": os.path.basename(path)})
    return out


def choose(results):
    """{arch: {size: micro}} plus the table that justifies it."""
    chosen, table, problems = {}, [], []
    for arch in ARCHES:
        field, why = RANK_BY[arch]
        for size in SIZES:
            here = [r for r in results if r["arch"] == arch and r["size"] == size]
            if not here:
                problems.append(f"{arch} {size}: nothing was measured")
                continue
            ranked = []
            for row in here:
                raw = row["fields"].get(field, "")
                failed = row["fields"].get("failed", "")
                if not raw:
                    table.append({"arch": arch, "size": size, "micro": row["micro"],
                                  "accum": row["accum"], field: None,
                                  "failed": failed or "no value in the line",
                                  "log": row["log"]})
                    continue
                if row["micro"] not in CANDIDATES:
                    problems.append(f"{arch} {size}: {row['micro']} is not a candidate")
                    continue
                if 256 % (row["micro"] * GPUS):
                    problems.append(f"{arch} {size}: {row['micro']} x {GPUS} GPUs "
                                    f"cannot build 256")
                    continue
                ranked.append((float(raw), row))
                table.append({"arch": arch, "size": size, "micro": row["micro"],
                              "accum": row["accum"], field: float(raw),
                              "failed": "", "log": row["log"]})
            if len(ranked) < 2:
                problems.append(f"{arch} {size}: {len(ranked)} usable shape(s); §4 "
                                f"asks for the fastest of several, not the only one "
                                f"that ran")
                continue
            ranked.sort(key=lambda pair: pair[0])
            best, runner_up = ranked[0], ranked[1]
            chosen.setdefault(arch, {})[size] = best[1]["micro"]
            table.append({"arch": arch, "size": size, "micro": best[1]["micro"],
                          "accum": best[1]["accum"], field: best[0], "failed": "",
                          "log": best[1]["log"], "chosen": True,
                          "margin_s": round(runner_up[0] - best[0], 3),
                          "runner_up_micro": runner_up[1]["micro"],
                          "ranked_by": field, "why": why})
    return chosen, table, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe-logs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write the file even where a size has no choice yet")
    args = ap.parse_args(argv)

    paths = sorted({p for pattern in args.probe_logs for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"no probe log matched {args.probe_logs}")
    results = parse(paths)
    chosen, table, problems = choose(results)

    print(f"=== {len(results)} measured shape(s) from {len(paths)} log(s) ===")
    for arch in ARCHES:
        field, _why = RANK_BY[arch]
        for size in SIZES:
            rows = sorted((r for r in table
                           if r["arch"] == arch and r["size"] == size
                           and not r.get("chosen")),
                          key=lambda r: r["micro"])
            if not rows:
                continue
            pick = chosen.get(arch, {}).get(size)
            cells = "  ".join(
                f"{r['micro']:>2}x{r['accum']:<4} "
                + (f"{r[field]:>7.3f}" if r[field] is not None else "  failed")
                + ("*" if r["micro"] == pick else " ")
                for r in rows)
            print(f"  {arch:5s} {size:7s} {cells}   -> {pick}")
    for problem in problems:
        print(f"  ** {problem}")
    if problems and not args.allow_incomplete:
        raise SystemExit("the choice is not complete; fix the above or pass "
                         "--allow-incomplete")

    payload = {
        "set": "measured-1120",
        "generated": _dt.datetime.now(JST).isoformat(timespec="seconds"),
        "directive": "2026-11-20 §4",
        "gpus": GPUS,
        "candidates": list(CANDIDATES),
        "ranked_by": {arch: {"field": RANK_BY[arch][0], "why": RANK_BY[arch][1]}
                      for arch in ARCHES},
        "chosen": chosen,
        "measurements": table,
        "logs": [os.path.basename(p) for p in paths],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nchosen: {json.dumps(chosen, sort_keys=True)}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
