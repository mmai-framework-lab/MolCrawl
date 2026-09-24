"""Every compounds GPT-2 run, read out of the logs the runs themselves wrote.

The 2026-09-24 order asks which learning rates were tried and what each reached.
Neither question can be answered from the configs: the sweep passed its learning
rate and its length at launch, so the config files say 6e-4 and 1,558 for runs
that trained at 1.6e-3 and 4,674. The launcher's own header line and nanoGPT's
per-evaluation line are what actually record a run.

  header  "size   : medium   lr: 1e-3   seed: 20"  (the launcher; later runs also
          carry "iters:", earlier ones do not)
  series  "step N: train loss X, val loss Y"       (nanoGPT, every evaluation)

Runs are grouped by the length they were given, because a 1,558-iteration run and
a 4,674-iteration one are not points on the same curve -- the shorter set stopped
while still improving, which is what the length probe established.

Read-only. Paths are arguments: this repository is public.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

STEP = re.compile(r"step\s+(\d+):\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)")
# Fields appear in a fixed order but not all runs carry all of them.
FIELD = re.compile(r"\b(size|lr|seed|iters)\s*:\s*(\S+)")
OUTPUT = re.compile(r"^outputs:\s*(\S+)", re.M)
HEADER = re.compile(r"^(?:code|size)\s*:.*$", re.M)


def read_run(path):
    """One run's identity and series, or None when the log holds no evaluation."""
    text = open(path, errors="ignore").read()
    head = "\n".join(HEADER.findall(text)[:3])
    ident = dict(FIELD.findall(head))
    out = OUTPUT.search(text)
    if out:
        ident.setdefault("run", os.path.basename(out.group(1)))
    pts = sorted((int(s), float(t), float(v)) for s, t, v in STEP.findall(text))
    return {
        "log": os.path.basename(path),
        "run": ident.get("run", ""),
        "size": ident.get("size", ""),
        "lr": ident.get("lr", ""),
        "seed": ident.get("seed", ""),
        # An older launcher did not print the length, and those runs are all the
        # 1,558-iteration set; say so rather than leaving the column empty.
        "iters_asked": ident.get("iters", ""),
        "series": pts,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", nargs="+", required=True,
                    help="glob(s) matching the run logs, e.g. '<dir>/mc-cmp-lrsweep-*.out'")
    ap.add_argument("--tsv", help="write every evaluation point here")
    args = ap.parse_args(argv)

    runs = []
    for pat in args.logs:
        for hit in sorted(glob.glob(pat)):
            runs.append(read_run(hit))

    width = max([len(r["run"]) for r in runs if r["run"]] + [20])
    print(f"{'run':<{width}} {'size':<7}{'lr':<9}{'seed':<6}{'asked':>7}{'reached':>9}"
          f"{'best_val':>10}{'@iter':>8}")
    for r in sorted(runs, key=lambda r: (r["size"], r["lr"], r["seed"])):
        if not r["series"]:
            print(f"{r['run'] or r['log']:<{width}} {r['size']:<7}{r['lr']:<9}{r['seed']:<6}"
                  f"{r['iters_asked'] or '-':>7}{0:>9}{'-':>10}{'-':>8}   (評価点なし)")
            continue
        best = min(r["series"], key=lambda p: p[2])
        print(f"{r['run'] or r['log']:<{width}} {r['size']:<7}{r['lr']:<9}{r['seed']:<6}"
              f"{r['iters_asked'] or '1558?':>7}{r['series'][-1][0]:>9}"
              f"{best[2]:>10.4f}{best[0]:>8}")

    reached = [r for r in runs if r["series"]]
    print(f"\nログ {len(runs)} 本、うち評価点を残したもの {len(reached)} 本")

    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as fh:
            fh.write("size\tlr\tseed\titers_asked\tstep\ttrain_loss\tval_loss\n")
            for r in sorted(runs, key=lambda r: (r["size"], r["lr"], r["seed"])):
                for step, tr, va in r["series"]:
                    fh.write(f"{r['size']}\t{r['lr']}\t{r['seed']}\t{r['iters_asked']}"
                             f"\t{step}\t{tr:.6f}\t{va:.6f}\n")
        n = sum(len(r["series"]) for r in runs)
        print(f"wrote {args.tsv} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
