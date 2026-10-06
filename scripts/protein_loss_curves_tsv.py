"""One TSV of every evaluation point for the protein_sequence BERT and GPT-2 runs.

boss protein-order-2026-10-06 (and the detailed protein-tsv-order-2026-10-06): one row
per evaluation point, all sizes, both architectures, as the actual data file.

Columns (protein-tsv-order-2026-10-06 §2):
    run  arch  size  lr  seed  step  value  metric  job  eval_set

Sources (no GPU; reads files only -- §5):
  * BERT: the newest checkpoint's trainer_state.json. HF restores the full log_history
    on resume, so one file carries every eval point across all segments. metric is
    eval_loss_mask on the fixed 10,000-row valid subset  -> eval_set=fixed.
    job per point comes from segments.log (segment -> job id) and the resume boundaries.
  * GPT-2: the logging_*.csv files (iter, train_loss, val_loss), one per segment, merged
    by iter (a later segment overrides). metric is val_loss. The 1.2e-3 runs set
    deterministic_val_eval -> eval_set=fixed; the retrain2 sweep redraws the subsample
    each eval -> eval_set=resampled. seed and job id come from run_manifest.json.

Run groups (protein-tsv-order-2026-10-06 §3): BERT grid 9, fill 9, large 3.5e-5 1,
seed-variance 6, xl 3  (= 28 present); GPT-2 retrain2 21, 1.2e-3 4 (= 25 present).
Groups 6 (xl upper 1.7e-5/2.4e-5/3.4e-5) and 7 (large 7e-5) are not launched.

Run:  python scripts/protein_loss_curves_tsv.py <out_dir>
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BERT = REPO / "learning_source_protein_runs/protein_sequence/bert-output"
GPT2_12E3 = REPO / "learning_source_protein_runs/protein_sequence/gpt2-output"
RETRAIN = REPO / "learning_source_20260730_protein_uniref50/protein_sequence/runs"

COLUMNS = ["run", "arch", "size", "lr", "seed", "step", "value", "metric", "job", "eval_set"]


def lrtag_to_val(tag: str) -> str:
    """BERT dir lr tag -> learning rate string. 4p1e4 -> 4.1e-4, 3p5e5 -> 3.5e-5, 1e3 -> 1e-3."""
    m = re.match(r"^(\d+(?:p\d+)?)e(\d+)$", tag)
    if not m:
        return tag
    return f"{m.group(1).replace('p', '.')}e-{m.group(2)}"


def manifest_seed(run_dir: Path):
    f = run_dir / "run_manifest.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    s = d.get("seed")
    if isinstance(s, dict):
        return s.get("seed")
    return s


def manifest_field(run_dir: Path, *path):
    f = run_dir / "run_manifest.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    for k in path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


# ---------- BERT ----------
def bert_segment_jobs(run_dir: Path):
    """Return (boundaries, seg_job[list]) so a step can be mapped to the job that ran it.

    segments.log rows are `segment jobid ISO-timestamp`. A continuation's start time tells
    us the newest checkpoint written before it; that checkpoint's step is the boundary.
    """
    seg_log = run_dir / "segments.log"
    seg_job = {}
    starts = []
    if seg_log.exists():
        import datetime as dt
        for line in open(seg_log):
            p = line.split()
            if len(p) >= 3:
                seg = int(p[0])
                seg_job[seg] = p[1]
                try:
                    starts.append((seg, dt.datetime.fromisoformat(p[2]).timestamp()))
                except ValueError:
                    pass
    ckpts = [(int(os.path.basename(p).split("-")[-1]), os.path.getmtime(p))
             for p in glob.glob(str(run_dir / "checkpoint-*"))]
    bounds = []
    for _seg, t in sorted(starts)[1:]:
        before = [s for s, m in ckpts if m <= t]
        if before:
            bounds.append(max(before))
    return sorted(set(bounds)), seg_job


def bert_rows(run_dir: Path):
    name = run_dir.name
    m = re.match(r"^protein_sequence-(small|medium|large|xl)-lr([0-9p]+e[0-9])(?:-s(\d+))?$", name)
    if not m:
        return None, f"unparsed BERT dir name: {name}"
    size, lrtag, sfx = m.group(1), m.group(2), m.group(3)
    lr = lrtag_to_val(lrtag)
    steps_ck = sorted((int(os.path.basename(p).split("-")[-1]) for p in glob.glob(str(run_dir / "checkpoint-*"))), reverse=True)
    if not steps_ck:
        return None, f"no checkpoint in {name}"
    ts = None
    for st in steps_ck:  # newest checkpoint that actually has a trainer_state (a live run may be mid-write)
        f = run_dir / f"checkpoint-{st}" / "trainer_state.json"
        if f.exists():
            ts = json.load(open(f))
            break
    if ts is None:
        return None, f"no trainer_state.json in any checkpoint of {name}"
    pts = sorted((e["step"], e["eval_loss_mask"]) for e in ts.get("log_history", []) if "eval_loss_mask" in e)
    if not pts:
        return None, f"no eval_loss_mask in {name}"
    # Resolved seed. The -sN suffix is the variance seed the run actually used (confirmed
    # against the run log and the manifest's data_seed); a few seed-variance manifests
    # misrecord the 'seed' field on resume, so the suffix wins when present.
    seed = int(sfx) if sfx else manifest_seed(run_dir)
    bounds, seg_job = bert_segment_jobs(run_dir)
    rows = []
    for step, val in pts:
        seg = 1 + sum(1 for b in bounds if step > b)
        job = seg_job.get(seg) or ";".join(seg_job.values()) or ""
        rows.append([name, "bert", size, lr, seed, step, f"{val:.6f}",
                     "eval_loss_mask", job, "fixed"])
    return rows, None


# ---------- GPT-2 ----------
def gpt2_merge_csv(run_dir: Path):
    pts = {}
    for csv in sorted(glob.glob(str(run_dir / "logging_*.csv"))):
        with open(csv) as fh:
            next(fh, None)
            for line in fh:
                p = line.split(",")
                if len(p) < 3:
                    continue
                try:
                    it = int(p[0])
                    vl = float(p[2])
                except ValueError:
                    continue
                pts[it] = vl  # a later segment overrides the same iter
    return sorted(pts.items())


def gpt2_retrain_rows(run_dir: Path):
    m = re.match(r"^retrain2-gpt2-(\w+?)-lr([\d.]+e-\d)-s(\d+)$", run_dir.name)
    if not m:
        return None, f"unparsed retrain2 name: {run_dir.name}"
    size, lr, seed_name = m.group(1), m.group(2), m.group(3)
    pts = gpt2_merge_csv(run_dir)
    if not pts:
        return None, f"no logging csv points in {run_dir.name}"
    seed = manifest_seed(run_dir) or int(seed_name)
    job = manifest_field(run_dir, "run", "job_id") or ""
    det = bool(manifest_field(run_dir, "eval", "deterministic_val_eval"))
    eval_set = "fixed" if det else "resampled"
    return [[run_dir.name, "gpt2", size, lr, seed, it, f"{v:.6f}", "val_loss", str(job), eval_set]
            for it, v in pts], None


def gpt2_12e3_rows(run_dir: Path):
    m = re.match(r"^protein_sequence-(small|medium|large|ex-large)-lr1p2e3$", run_dir.name)
    if not m:
        return None, f"unparsed 1.2e-3 name: {run_dir.name}"
    size = "xl" if m.group(1) == "ex-large" else m.group(1)
    pts = gpt2_merge_csv(run_dir)
    if not pts:
        return None, f"no logging csv points in {run_dir.name}"
    seed = manifest_seed(run_dir)
    job = manifest_field(run_dir, "run", "job_id") or ""
    det = bool(manifest_field(run_dir, "eval", "deterministic_val_eval"))
    eval_set = "fixed" if det else "resampled"
    return [[run_dir.name, "gpt2", size, "1.2e-3", seed, it, f"{v:.6f}", "val_loss", str(job), eval_set]
            for it, v in pts], None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/protein_loss_curves_tsv.py <out_dir>")
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "protein-loss-curves.tsv"

    all_rows = []
    per_run = {}
    problems = []

    # BERT (28 expected)
    for d in sorted(glob.glob(str(BERT / "protein_sequence-*"))):
        d = Path(d)
        if not d.is_dir():
            continue
        rows, err = bert_rows(d)
        if err:
            problems.append(err)
            continue
        all_rows += rows
        per_run[d.name] = len(rows)

    # GPT-2 retrain2 (21 expected)
    for d in sorted(glob.glob(str(RETRAIN / "retrain2-gpt2-*"))):
        d = Path(d)
        if not d.is_dir():
            continue
        rows, err = gpt2_retrain_rows(d)
        if err:
            problems.append(err)
            continue
        all_rows += rows
        per_run[d.name] = len(rows)

    # GPT-2 1.2e-3 (4 expected)
    for d in sorted(glob.glob(str(GPT2_12E3 / "protein_sequence-*-lr1p2e3"))):
        d = Path(d)
        if not d.is_dir():
            continue
        rows, err = gpt2_12e3_rows(d)
        if err:
            problems.append(err)
            continue
        all_rows += rows
        per_run[d.name] = len(rows)

    with open(tsv, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in all_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # ---- verification (§4) ----
    bert_runs = [r for r in per_run if r.startswith("protein_sequence-") and "lr1p2e3" not in r]
    g_retrain = [r for r in per_run if r.startswith("retrain2-gpt2-")]
    g_12e3 = [r for r in per_run if r.endswith("lr1p2e3")]
    no_job = sorted({r[0] for r in all_rows if not r[8]})
    missing_seed = sorted({r[0] for r in all_rows if r[4] in (None, "", "None")})

    print(f"wrote {tsv}  ({len(all_rows)} rows)")
    print(f"BERT runs: {len(bert_runs)} (expect 28)")
    print(f"GPT-2 retrain2 runs: {len(g_retrain)} (expect 21)")
    print(f"GPT-2 1.2e-3 runs: {len(g_12e3)} (expect 4)")
    print(f"total runs: {len(per_run)} (expect 53; groups 6 xl-upper and 7 large-7e5 NOT launched)")
    print(f"rows with no job: {len(no_job)} {no_job}")
    print(f"runs with no seed: {missing_seed}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  ", p)
    # distinct eval_set per arch
    es = {}
    for r in all_rows:
        es.setdefault((r[1], r[9]), set()).add(r[0])
    print("eval_set coverage:")
    for (arch, eset), runs in sorted(es.items()):
        print(f"  {arch} {eset}: {len(runs)} runs")


if __name__ == "__main__":
    main()
