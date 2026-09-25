"""Collect every genome evaluation point into one TSV, series by series.

The five series are not one experiment. Their windows differ (512, 1,024,
1,026), their epoch counts differ (3 and 9), and until now their checkpoint
selection differed too -- the 512-window BERT runs were judged on ``eval_loss``
and the 1,026-window ones on ``eval_loss_mask``. So this writes one row per
evaluation point with the series named on it, and never merges two series into
one curve.

**The metric is pinned, not inherited.** Every BERT row is ``eval_loss_mask``,
read from ``log_history`` rather than from whichever metric that run selected
on, so the five BERT series are on one scale and on the same scale as rna,
molecule_nat_lang and compounds (``judge_on`` defaults to it in
``models/bert/main.py``). GPT-2 has no masking, so its rows are ``val_loss``,
and the two are labelled per row rather than assumed from the series name.

**Nothing is thinned.** A curve read off decimated points has the wrong shape
exactly where it matters, at the turn. The 512-window series evaluates every
100 steps and the others every 1,000, so point counts differ by a factor of
ten between series -- that is a property of the runs, not something to even out
here.

Job numbers come from four different places because the runs were launched
across two months and the manifest only exists for the later ones. Each is
derived, none is typed in: see ``job_id_for``.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess

BERT_METRIC = "eval_loss_mask"
GPT2_METRIC = "val_loss"


def series_of(basename):
    """Which series a run directory belongs to, by its name suffix."""
    for suffix, name in (("-w1026", "bert-w1026"), ("-sat9", "bert-sat9"),
                         ("-w1024-ep9", "bert-w1024-ep9"), ("-smoke", "smoke")):
        if basename.endswith(suffix):
            return name
    return "bert-base"          # the 512-window production series, untagged


def subset_of(basename):
    name = re.sub(r"^(bert|gpt2)-small-", "", basename)
    return re.sub(r"-(w1026|sat9|w1024-ep9|smoke)$", "", name)


def _sacct_rows(start, end, user):
    """One row per job in the window: (JobID, JobName, NodeList)."""
    try:
        out = subprocess.run(
            ["sacct", "-S", start, "-E", end, "-u", user, "-X", "-P", "-n",
             "-o", "JobID,JobName,NodeList"],
            capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.split("|") for line in out.strip().splitlines() if line]


def job_id_for(run_dir, series, log_dir, sacct):
    """The Slurm job that produced this run.

    Four sources, because the runs span two months of changing conventions:
    the manifest (written only for the 1,026-window runs and the 1,024 probe),
    the slurm log file name (the 512-window series), the log name again for
    sat9 whose manifest recorded an empty LEARNING_SOURCE_DIR, and for GPT-2 --
    which writes no manifest and whose logs do not carry the subset -- sacct
    matched on the node name that tensorboard baked into its event file.
    """
    base = os.path.basename(run_dir)
    manifest = os.path.join(run_dir, "run_manifest.json")
    if os.path.exists(manifest):
        src = json.load(open(manifest)).get("env", {}).get("LEARNING_SOURCE_DIR", "")
        hit = re.search(r"genome-bert-(\d+)", src)
        if hit:
            return hit.group(1)

    if series.startswith("bert"):
        pattern = re.compile(rf"-{re.escape(subset_of(base))}-(\d+)\.out$")
        for name in sorted(os.listdir(log_dir) if os.path.isdir(log_dir) else []):
            hit = pattern.search(name)
            if hit and ("sat9" in name) == (series == "bert-sat9"):
                return hit.group(1)
        return ""

    events = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not events:
        return ""
    node = os.path.basename(events[0]).rsplit(".", 1)[1]
    hits = [r[0] for r in sacct if r[2] == node and "gpt2" in r[1]]
    return hits[0] if len(hits) == 1 else ""


def bert_meta(run_dir):
    """Window, schedule and batch shape, read from what the run saved.

    Every figure has to state these beside its curves, and a caption typed by
    hand drifts from the run it describes. `epochs` is recomputed rather than
    taken from trainer_state's `num_train_epochs`, which HF rounds up whenever
    the last epoch is partial -- it reads 4 on a 3-epoch run and 10 on a 9.
    """
    import torch
    steps = [int(p.rsplit("-", 1)[1]) for p in glob.glob(f"{run_dir}/checkpoint-*")
             if p.rsplit("-", 1)[1].isdigit()]
    if not steps:
        return {}
    last = f"{run_dir}/checkpoint-{max(steps)}"
    args = torch.load(f"{last}/training_args.bin", map_location="cpu", weights_only=False)
    window = json.load(open(f"{last}/config.json")).get("max_position_embeddings")
    batch = (args.per_device_train_batch_size * args.gradient_accumulation_steps
             * args.world_size)
    state = json.load(open(f"{last}/trainer_state.json"))
    evals = [e["step"] for e in state["log_history"] if BERT_METRIC in e]
    return {"window": window, "max_steps": args.max_steps, "global_batch": batch,
            "lr": args.learning_rate, "warmup": args.warmup_steps,
            "eval_every": (evals[1] - evals[0]) if len(evals) > 1 else None,
            "epochs": _manifest_epochs(run_dir)}


def _manifest_epochs(run_dir):
    """The epoch count the run was launched with, or None.

    Only read from the manifest, never inferred from max_steps. max_steps is
    ceil(epochs x train_rows / batch) and train_rows is not in the checkpoint,
    so any rule that recovers epochs from max_steps alone is a threshold fitted
    to the runs that happen to exist -- it would answer confidently and wrongly
    the first time a different length is launched. The twenty-one 512-window
    runs predate the manifest and report nothing here; their length is stated
    in the report, from the arithmetic against that build's train row count.
    """
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(path):
        return None
    value = json.load(open(path)).get("env", {}).get("SUBSET_BERT_EPOCHS")
    return int(value) if value and str(value).isdigit() else None


def gpt2_meta(run_dir):
    """The same, from the nanoGPT checkpoint's saved config."""
    import torch
    path = os.path.join(run_dir, "ckpt.pt")
    if not os.path.exists(path):
        return {}
    cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
    return {"window": cfg.get("block_size"), "max_steps": cfg.get("max_iters"),
            "global_batch": cfg.get("batch_size") * cfg.get("gradient_accumulation_steps"),
            "lr": cfg.get("learning_rate"), "warmup": cfg.get("warmup_iters"),
            # _N_EPOCH is a literal 3 in gpt2_small_subset.py, not an env
            # override, so the config file is the record for GPT-2.
            "eval_every": cfg.get("eval_interval"), "epochs": 3}


def bert_points(run_dir):
    """Every eval_loss_mask point, from the newest checkpoint's trainer_state."""
    steps = [int(p.rsplit("-", 1)[1]) for p in glob.glob(f"{run_dir}/checkpoint-*")
             if p.rsplit("-", 1)[1].isdigit()]
    if not steps:
        return []
    state = json.load(open(f"{run_dir}/checkpoint-{max(steps)}/trainer_state.json"))
    return [(e["step"], float(e[BERT_METRIC]))
            for e in state["log_history"] if BERT_METRIC in e]


def gpt2_points(run_dir):
    """Every val_loss point, from the run's non-empty logging CSV.

    Nine of the 21 runs hold two CSVs, one second apart: two ranks each opened
    a file and only one wrote. They are not resumed segments -- there is no
    overlap and the written one starts at iter 0 -- so taking whichever file
    sorts first would silently return nothing for five runs.
    """
    rows = []
    for path in sorted(glob.glob(f"{run_dir}/logging_*.csv")):
        with open(path) as handle:
            got = [(int(r["iter"]), float(r[GPT2_METRIC]))
                   for r in csv.DictReader(handle, skipinitialspace=True)
                   if r.get("iter", "").strip().isdigit()]
        if got:
            rows.append(got)
    if len(rows) != 1:
        return rows[0] if rows else []
    return rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--log-dir", required=True, help="workflows/slurm-logs")
    ap.add_argument("--sacct-start", default="2026-08-16")
    ap.add_argument("--sacct-end", default="2026-08-20")
    ap.add_argument("--user", default=os.environ.get("USER", ""))
    ap.add_argument("--out", required=True, help="the TSV every figure is drawn from")
    ap.add_argument("--meta-out", default="", help="per-series metadata the captions are built from")
    args = ap.parse_args()

    sacct = _sacct_rows(args.sacct_start, args.sacct_end, args.user)
    print(f"  sacct rows in window: {len(sacct)}", flush=True)

    rows, skipped, meta = [], [], {}
    for run_dir in sorted(glob.glob(f"{args.runs_root}/bert-small-*")
                          + glob.glob(f"{args.runs_root}/gpt2-small-*")):
        base = os.path.basename(run_dir)
        arch = "bert" if base.startswith("bert-") else "gpt2"
        series = series_of(base) if arch == "bert" else "gpt2"
        if series == "smoke":                      # 200 steps, excluded by the order
            continue
        points = bert_points(run_dir) if arch == "bert" else gpt2_points(run_dir)
        if not points:
            skipped.append(base)
            continue
        job = job_id_for(run_dir, series, args.log_dir, sacct)
        got = bert_meta(run_dir) if arch == "bert" else gpt2_meta(run_dir)
        if got:
            slot = meta.setdefault(series, {})
            for key, value in got.items():
                slot.setdefault(key, set()).add(value)
        metric = BERT_METRIC if arch == "bert" else GPT2_METRIC
        for step, value in points:
            rows.append((series, arch, subset_of(base), step, f"{value:.6f}", metric, job))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["series", "arch", "subset", "step", "value", "metric", "job"])
        writer.writerows(rows)

    if args.meta_out:
        # Ranges, not single values: max_steps and train_rows differ per subset
        # by design, and a caption that prints one of them as the series' value
        # would be wrong for twenty runs out of twenty-one.
        shaped = {s: {k: (sorted(v)[0] if len(v) == 1 else [min(v), max(v)])
                      for k, v in d.items() if None not in v}
                  for s, d in meta.items()}
        json.dump(shaped, open(args.meta_out, "w"), indent=2, ensure_ascii=False)
        print(f"  wrote {args.meta_out}")

    by_series = {}
    for series, _arch, subset, _s, _v, _m, job in rows:
        got = by_series.setdefault(series, {"runs": set(), "points": 0, "jobs": set()})
        got["runs"].add(subset)
        got["points"] += 1
        got["jobs"].add(job)
    print(f"\n  {'series':16s} {'runs':>5s} {'points':>8s}  job 番号の取れた run")
    for series in sorted(by_series):
        got = by_series[series]
        named = len([j for j in got["jobs"] if j])
        print(f"  {series:16s} {len(got['runs']):>5d} {got['points']:>8,}  "
              f"{named}/{len(got['jobs'])} 種")
    if skipped:
        print(f"\n  評価点が取れなかった run: {skipped}")
    print(f"\n  wrote {args.out}  ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
