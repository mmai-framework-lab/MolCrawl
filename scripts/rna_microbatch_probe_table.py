#!/usr/bin/env python
"""Read the rna micro-batch probe's logs and say which one-GPU block count to use.

§4 of the rna batch-size directive of 2026-11-20 asks for the step time measured per
size and the chosen value reported; §9 wants the block count and accumulation of every
arm. This turns the probe's 31 logs into both, and into the table to paste into
scripts/rna_batch_sweep_arms.py as the 'measured' set.

    rna_microbatch_probe_table.py --record <submit record TSV> --logs <slurm-logs dir>

The record names every arm and its job id; the log of job N is the file whose name
ends in -N.out. An arm that does not appear, or that reports FAILED, is carried as
such rather than dropped: "it does not fit" is an answer about the shape, and a
missing row would silently change which value looks fastest.

Nothing here reruns an arm. It reads finished logs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

#: BERT prints "  <label>: 4.482 s/step  (steps 50-300)"; GPT-2 prints
#: "VERDICT <label>: OK  0.954 s/iter  (251 samples from iter 50)". Both also print
#: FAILED with a reason, and both can report no usable timing samples.
BERT_RATE = re.compile(r"^\s+\S+:\s+([\d.]+)\s+s/step\b", re.M)
GPT2_RATE = re.compile(r"^VERDICT\s+\S+:\s+OK\s+([\d.]+)\s+s/iter\b", re.M)
FAILED = re.compile(r"FAILED \(([^)]*)\)")
NO_SAMPLES = re.compile(r"no usable timing samples")

#: The grid's five batches, for the accumulation table of §9.
BATCHES = [256, 512, 1024, 2048, 4096]
#: BERT multiplies by world_size, nanoGPT divides the accumulation across the ranks,
#: so both need micro x 4 to divide the batch. Same asymmetry as the arms script.
GPU_COUNT = 4
#: What the arms run today, so the table can say what staying put costs.
CURRENT = {"bert": {"small": 8, "medium": 8, "large": 8, "xl": 8},
           "gpt2": {"small": 16, "medium": 16, "large": 8, "xl": 4}}
SIZES = ["small", "medium", "large", "xl"]
REFERENCE_BATCH = 2560


def read_record(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle, delimiter="\t") if r.get("job_id")]


def log_for(job_id: str, logs_dir: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(logs_dir, f"*-{job_id}.out")))
    return hits[-1] if hits else None


def rate_from(text: str, arch: str) -> tuple[float | None, str]:
    """Seconds per step at the probe's global batch, and what the log says."""
    failed = FAILED.search(text)
    if failed:
        return None, f"FAILED ({failed.group(1)})"
    pattern = BERT_RATE if arch == "bert" else GPT2_RATE
    hit = pattern.search(text)
    if hit:
        return float(hit.group(1)), "OK"
    if NO_SAMPLES.search(text):
        return None, "no usable timing samples"
    return None, "no verdict in the log"


def read_table(path: str) -> list[dict]:
    """Rows of a table this script wrote earlier."""
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(
            (ln for ln in handle if not ln.startswith("#")), delimiter="\t"))


def fastest_micro(rows: list[dict]) -> dict[str, dict[str, int]]:
    """The one-GPU block count to run, per framework and size: the fastest measured.

    Ties go to the smaller block count, which needs less memory for the same speed.
    A size with no measured arm is left out rather than guessed, so a caller that
    needs all four finds out instead of quietly running a value nobody measured.
    """
    best: dict[str, dict[str, int]] = {}
    for row in rows:
        rate = row.get("s_per_step_at_2560") or ""
        if not rate:
            continue
        arch, size, micro = row["arch"], row["size"], int(row["micro"])
        seat = best.setdefault(arch, {})
        if size not in seat or (float(rate), micro) < seat[size][0]:
            seat[size] = ((float(rate), micro), micro)
    return {arch: {size: pick for size, (_, pick) in sizes.items()}
            for arch, sizes in best.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", required=True, help="the probe's submit record TSV")
    ap.add_argument("--logs", required=True, help="directory holding the arms' logs")
    ap.add_argument("--out", help="TSV to write the per-arm measurements to")
    args = ap.parse_args()

    rows = []
    for arm in read_record(args.record):
        path = log_for(arm["job_id"], args.logs)
        if path is None:
            rate, status = None, "no log (did the arm run?)"
        else:
            with open(path, errors="ignore", encoding="utf-8") as handle:
                rate, status = rate_from(handle.read(), arm["arch"])
        glob_batch = int(arm["global"])
        rows.append({
            "arch": arm["arch"], "size": arm["size"], "micro": int(arm["micro"]),
            "global": glob_batch, "accum": int(arm["accum"]), "job_id": arm["job_id"],
            "status": status,
            "s_per_step": f"{rate:.3f}" if rate else "",
            "s_per_step_at_2560": (f"{rate * REFERENCE_BATCH / glob_batch:.3f}"
                                   if rate else ""),
            "log": os.path.basename(path) if path else "",
        })

    done = [r for r in rows if r["s_per_step"]]
    print(f"=== {len(rows)} arms in the record, {len(done)} with a rate,"
          f" {len(rows) - len(done)} without")
    for r in rows:
        if not r["s_per_step"]:
            print(f"    {r['arch']} {r['size']} micro {r['micro']}:"
                  f" {r['status']} (job {r['job_id']})")

    # One rule for the choice, used here and by the arms script.
    chosen = fastest_micro(rows)
    for arch in ("bert", "gpt2"):
        print(f"\n== {arch}: s/step at global 2,560 (measured at the probe's batch)")
        header = f"   {'size':<8}" + "".join(f"{f'micro {m}':>12}"
                                             for m in (4, 8, 16, 32, 64))
        print(header + f"{'fastest':>10}{'現行比':>9}")
        for size in SIZES:
            cells, by_micro = [], {}
            for micro in (4, 8, 16, 32, 64):
                hit = [r for r in done
                       if r["arch"] == arch and r["size"] == size
                       and r["micro"] == micro]
                if not hit:
                    bad = [r for r in rows
                           if r["arch"] == arch and r["size"] == size
                           and r["micro"] == micro]
                    cells.append(f"{'OOM' if bad else '-':>12}")
                    continue
                value = float(hit[0]["s_per_step_at_2560"])
                by_micro[micro] = value
                cells.append(f"{value:>12.2f}")
            if not by_micro:
                print(f"   {size:<8}" + "".join(cells) + f"{'-':>10}{'-':>9}")
                continue
            best = chosen[arch][size]
            now = CURRENT[arch][size]
            ratio = (by_micro[now] / by_micro[best]) if now in by_micro else None
            print(f"   {size:<8}" + "".join(cells)
                  + f"{f'micro {best}':>10}"
                  + (f"{ratio:>8.2f}x" if ratio else f"{'-':>9}"))

    if chosen:
        print("\n== the 'measured' set for scripts/rna_batch_sweep_arms.py")
        print('    "measured": {')
        for arch in ("bert", "gpt2"):
            if arch not in chosen:
                continue
            body = ", ".join(f'"{s}": {chosen[arch][s]}'
                             for s in SIZES if s in chosen[arch])
            print(f'        "{arch}": {{{body}}},')
        print("    },")

        print("\n== §9: accumulation at the chosen block count")
        print(f"   {'arch':<6}{'size':<8}{'micro':>7}"
              + "".join(f"{f'b{b}':>9}" for b in BATCHES))
        for arch in ("bert", "gpt2"):
            for size in SIZES:
                micro = chosen.get(arch, {}).get(size)
                if micro is None:
                    continue
                cells = []
                for batch in BATCHES:
                    unit = micro * GPU_COUNT
                    cells.append(f"{batch // unit:>9,}" if batch % unit == 0
                                 else f"{'不可':>9}")
                print(f"   {arch:<6}{size:<8}{micro:>7}" + "".join(cells))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("#! report-tsv 1\n")
            handle.write("# model: several (columns: arch,size)\n")
            handle.write("# note: the rna micro-batch probe of 2026-11-20, §4 of the"
                         " batch-size directive. One arm per job, 300 steps, cold"
                         " cache, measured at the global batch in the 'global'"
                         " column and converted to 2,560 for comparison.\n")
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")

    missing = [r for r in rows if not r["s_per_step"] and "FAILED" not in r["status"]]
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
