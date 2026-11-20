"""Read the per-step time of one batch-shape probe out of its own log.

§4 of the 2026-11-20 directive asks for one GPU's block count to be chosen per size by
measurement, and for the measured numbers to be reported. This is the reading step, so
that what gets reported is taken from the run's own output rather than from a stopwatch
or from a remembered figure.

    molnl_batch_shape_timing.py --arch bert --steps 40 <log>
    molnl_batch_shape_timing.py --arch gpt2 --steps 60 --warmup 20 <log>

The two frameworks say it differently.

BERT (HF Trainer) prints one summary dict at the end, with train_runtime over the whole
call and eval_runtime for each evaluation. The training time is train_runtime minus the
evaluations inside it, and the per-step figure is that over the step count. A probe is
run with one evaluation at the end for this reason: more of them and the subtraction
carries more of their variance than of the thing being measured.

GPT-2 (nanoGPT) prints every iteration with its own time in milliseconds, so the
reading is a mean over the iterations after the warmup -- the first ones carry the
allocator growing its pools and the kernels being chosen, which is startup, not
throughput. The same cannot be done for BERT: the Trainer does not log a per-step time.

Prints one line of ``key=value`` pairs, which is what the probe job collects into its
table, and exits non-zero when the log does not hold what it needs -- an interrupted or
failed probe must not be read as a fast one.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys

#: HF's end-of-training dict, e.g. {'train_runtime': 7229.8683, 'train_samples_...
TRAIN_RUNTIME = re.compile(r"'train_runtime':\s*([0-9.]+)")
EVAL_RUNTIME = re.compile(r"'eval_runtime':\s*([0-9.]+)")
#: nanoGPT's per-iteration line, e.g. "iter 12: loss 7.1234, time 3456.78ms, mfu 1.23%"
NANOGPT_ITER = re.compile(r"^iter\s+(\d+):.*?time\s+([0-9.]+)ms")


def read_bert(text, steps):
    train = [float(m) for m in TRAIN_RUNTIME.findall(text)]
    if not train:
        raise SystemExit("no train_runtime in the log: the probe did not finish")
    evals = [float(m) for m in EVAL_RUNTIME.findall(text)]
    runtime = train[-1]
    eval_total = sum(evals)
    per_step = (runtime - eval_total) / steps
    return {
        "per_step_s": f"{per_step:.3f}",
        "train_runtime_s": f"{runtime:.1f}",
        "eval_runtime_s": f"{eval_total:.1f}",
        "eval_points": len(evals),
        "steps": steps,
        "source": "HF train_runtime less the evaluations inside it",
    }


def read_gpt2(text, steps, warmup):
    times = [(int(m.group(1)), float(m.group(2)) / 1000.0)
             for m in (NANOGPT_ITER.match(line) for line in text.splitlines()) if m]
    if not times:
        raise SystemExit("no per-iteration lines in the log: the probe did not run")
    kept = [t for it, t in times if it >= warmup]
    if len(kept) < 5:
        raise SystemExit(f"only {len(kept)} iterations past the {warmup}-iteration "
                         f"warmup; the probe is too short to read")
    per_step = statistics.mean(kept)
    return {
        "per_step_s": f"{per_step:.3f}",
        "median_s": f"{statistics.median(kept):.3f}",
        "stdev_s": f"{statistics.stdev(kept):.3f}" if len(kept) > 1 else "",
        "iters_read": len(kept),
        "iters_logged": len(times),
        "warmup": warmup,
        "source": "mean of the per-iteration times nanoGPT logs, past the warmup",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--arch", required=True, choices=("bert", "gpt2"))
    ap.add_argument("--steps", type=int, required=True,
                    help="the step count the probe was given (BERT: the divisor)")
    ap.add_argument("--warmup", type=int, default=None,
                    help="GPT-2: iterations to drop before averaging (default: a third)")
    ap.add_argument("--label", default="", help="echoed back in the line")
    args = ap.parse_args(argv)

    with open(args.log, encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    if args.arch == "bert":
        values = read_bert(text, args.steps)
    else:
        warmup = args.warmup if args.warmup is not None else max(5, args.steps // 3)
        values = read_gpt2(text, args.steps, warmup)

    parts = [f"arch={args.arch}"]
    if args.label:
        parts.append(f"label={args.label}")
    parts += [f"{k}={v}" for k, v in values.items()]
    print("SHAPE_RESULT " + " ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
