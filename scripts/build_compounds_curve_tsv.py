"""One TSV for every compounds loss curve, BERT and GPT-2 in the same eight columns.

The figures are drawn from this file, so the numbers behind a curve can be read
without trusting the picture. Two record formats feed it and they are not the same
measurement, which is why the metric is a column rather than an assumption:

  bert  log_history in checkpoint-*/trainer_state.json; the adoption metric is
        eval_loss_mask, the [MASK] positions alone.
  gpt2  "step N: train loss X, val loss Y" on nanoGPT's stdout; val_loss is the
        whole held-out slice, with no masking involved.

The job number is carried per run because it is the only handle on what a run was
actually launched with. The compounds configs have been overridden at launch
before -- run 53767 trained at 15,000 steps and 1e-3 while its config said 1,558
and 1e-4 -- so a row that cannot be traced back to a job cannot be checked.

Read-only. Paths and job numbers are arguments: this repository is public.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

STEP = re.compile(r"step\s+(\d+):\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)")
FIELD = re.compile(r"\b(size|lr|seed|iters)\s*:\s*(\S+)")
HEADER = re.compile(r"^(?:code|size)\s*:.*$", re.M)
COLUMNS = ["arch", "size", "lr", "seed", "step", "value", "metric", "job"]


def bert_rows(spec, metric):
    """spec: five values -- run dir, size, lr, seed, job.

    Five separate values rather than one delimited string because the launcher
    passes this through sbatch --export, whose own separator is a comma: a
    "dir=...,size=..." spec is silently truncated at the first comma there.
    """
    f = dict(zip(("dir", "size", "lr", "seed", "job"), spec))
    cks = glob.glob(os.path.join(f["dir"], "checkpoint-*", "trainer_state.json"))
    if not cks:
        raise SystemExit(f"no checkpoint under {f['dir']}")
    newest = max(cks, key=lambda p: int(os.path.basename(os.path.dirname(p)).split("-")[1]))
    state = json.load(open(newest))
    out = []
    for e in state["log_history"]:
        if metric in e:
            out.append({"arch": "bert", "size": f["size"], "lr": f["lr"], "seed": f["seed"],
                        "step": e["step"], "value": f"{e[metric]:.6f}", "metric": metric,
                        "job": f["job"]})
    if not out:
        raise SystemExit(f"{f['dir']} records no {metric}")
    return out


def gpt2_rows(pattern, iters_only):
    """Every nanoGPT log matching the pattern, with the job number from its name."""
    out = []
    for path in sorted(glob.glob(pattern)):
        text = open(path, errors="ignore").read()
        ident = dict(FIELD.findall("\n".join(HEADER.findall(text)[:3])))
        if iters_only and ident.get("iters") != iters_only:
            continue
        job = re.search(r"(\d+)\.out$", path)
        for step, _train, val in STEP.findall(text):
            out.append({"arch": "gpt2", "size": ident.get("size", ""), "lr": ident.get("lr", ""),
                        "seed": ident.get("seed", ""), "step": int(step), "value": f"{float(val):.6f}",
                        "metric": "val_loss", "job": job.group(1) if job else ""})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bert", action="append", default=[], nargs=5,
                    metavar=("DIR", "SIZE", "LR", "SEED", "JOB"),
                    help="one BERT run: its directory, size, learning rate, seed, job (repeatable)")
    ap.add_argument("--bert-metric", default="eval_loss_mask")
    ap.add_argument("--gpt2-logs", help="glob for the nanoGPT logs")
    ap.add_argument("--gpt2-iters", help="keep only runs launched with this length")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = []
    for spec in args.bert:
        rows.extend(bert_rows(spec, args.bert_metric))
    if args.gpt2_logs:
        rows.extend(gpt2_rows(args.gpt2_logs, args.gpt2_iters))

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in sorted(rows, key=lambda r: (r["arch"], r["size"], r["lr"], r["seed"], r["step"])):
            fh.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")

    seen = {}
    for r in rows:
        seen.setdefault((r["arch"], r["size"], r["lr"], r["seed"], r["job"]), 0)
        seen[(r["arch"], r["size"], r["lr"], r["seed"], r["job"])] += 1
    print(f"wrote {args.out}: {len(rows)} rows, {len(seen)} runs")
    for k, n in sorted(seen.items()):
        print(f"  {k[0]:<5}{k[1]:<7}lr {k[2]:<8}seed {k[3]:<6}job {k[4]:<8}{n:>5} 点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
