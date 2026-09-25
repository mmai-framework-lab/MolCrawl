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
import datetime
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
    """One row per job in the window: (JobID, JobName, NodeList, Start, End)."""
    try:
        out = subprocess.run(
            ["sacct", "-S", start, "-E", end, "-u", user, "-X", "-P", "-n",
             "-o", "JobID,JobName,NodeList,Start,End"],
            capture_output=True, text=True, timeout=180).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        field = line.split("|")
        if len(field) < 5:
            continue
        try:
            began = datetime.datetime.fromisoformat(field[3])
            ended = datetime.datetime.fromisoformat(field[4])
        except ValueError:
            continue                       # PENDING and CANCELLED carry "Unknown"
        rows.append((field[0], field[1], field[2], began, ended))
    return rows


def _log_headers(log_dir):
    """Every `job=<id> subset=<name>` header the run scripts print.

    The file name is not enough: the two 21-run BERT series produce jobs of the
    same name for the same subset, months apart, and some jobs carry no subset
    in their name at all. The header inside the log states both.
    """
    pattern = re.compile(r"job=(\d+)\s+subset=([A-Za-z0-9_]+)")
    found = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.out"))):
        try:
            with open(path, errors="replace") as handle:
                for _ in range(5):
                    hit = pattern.search(handle.readline())
                    if hit:
                        found.append((hit.group(1), hit.group(2)))
                        break
        except OSError:
            continue
    return found


def _run_mtime(run_dir):
    """When this run last wrote -- used to tell two attempts of one subset apart."""
    checkpoints = [p for p in glob.glob(f"{run_dir}/checkpoint-*")
                   if p.rsplit("-", 1)[1].isdigit()]
    path = (max(checkpoints, key=lambda p: int(p.rsplit("-", 1)[1]))
            if checkpoints else os.path.join(run_dir, "ckpt.pt"))
    if not os.path.exists(path):
        return None
    return datetime.datetime.fromtimestamp(os.path.getmtime(path))


def job_id_for(run_dir, series, headers, sacct):
    """The Slurm job that produced this run, or "" when it cannot be pinned.

    Matching on the log file name alone is wrong here and quietly so: the two
    21-run BERT series produce identically named jobs for the same subset two
    weeks apart, so the untagged 512-window run picks up the 1,026-window run's
    number and nothing complains. Both of the routes below therefore intersect
    an identity with the window the job actually ran in, checked against when
    the run last wrote a checkpoint.

    BERT: the subset comes from the `job=<id> subset=<name>` header the run
    script prints. GPT-2 writes no such header and no manifest, so its identity
    is the node name tensorboard baked into the event file.

    More than one candidate returns "" rather than the first. A wrong job
    number in the table is worse than a missing one -- it reads as a fact.
    """
    when = _run_mtime(run_dir)
    if when is None:
        return ""
    ran = {row[0] for row in sacct
           if row[3] <= when <= row[4] + datetime.timedelta(minutes=30)}

    if series.startswith("bert"):
        subset = subset_of(os.path.basename(run_dir))
        hits = sorted({job for job, name in headers if name == subset and job in ran})
    else:
        events = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
        if not events:
            return ""
        node = os.path.basename(events[0]).rsplit(".", 1)[1]
        hits = sorted({row[0] for row in sacct
                       if row[2] == node and "gpt2" in row[1] and row[0] in ran})
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


def _measured_epochs(root, subset, max_steps, batch, model):
    """max_steps x batch / train rows, accepted only if it lands on an integer.

    The row count is taken by opening the split, not from dataset_info.json,
    which reports 95,016,076 where the split holds 94,916,076 -- a hundred
    thousand rows out, and the source of a step-count error corrected on
    2026-08-24. The integer check is what makes this safe to run against a root
    that might be the wrong build: a mismatched root gives a fractional answer
    and the field stays empty rather than gaining a plausible wrong number.
    """
    path = os.path.join(root, subset, f"training_ready_hf_dataset_{model}")
    if not os.path.isdir(path):
        return None
    try:
        from datasets import load_from_disk
        rows = len(load_from_disk(path)["train"])
    except Exception:
        return None
    epochs = max_steps * batch / rows
    return round(epochs) if abs(epochs - round(epochs)) < 0.005 else None


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
    ap.add_argument("--train-root", nargs="*", default=[],
                    help="series=genome_sequence root, to count that build's train "
                         "rows and turn max_steps into an epoch count")
    ap.add_argument("--out", required=True, help="the TSV every figure is drawn from")
    ap.add_argument("--meta-out", default="", help="per-series metadata the captions are built from")
    args = ap.parse_args()

    sacct = _sacct_rows(args.sacct_start, args.sacct_end, args.user)
    headers = _log_headers(args.log_dir)
    print(f"  sacct rows in window: {len(sacct)}   log headers: {len(headers)}", flush=True)
    train_root = dict(item.partition("=")[::2] for item in args.train_root)

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
        job = job_id_for(run_dir, series, headers, sacct)
        got = bert_meta(run_dir) if arch == "bert" else gpt2_meta(run_dir)
        if got and got.get("epochs") is None and series in train_root:
            got["epochs"] = _measured_epochs(
                train_root[series], subset_of(base), got["max_steps"], got["global_batch"],
                "bert" if arch == "bert" else "gpt2")
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
