"""What happened to the jobs a session submitted, read back from Slurm.

A session that submits long runs and then ends leaves nobody watching them. Four
compounds BERT runs died on a Friday evening and were not noticed until Monday --
56 hours with nothing of that workstream running -- because the only watcher was a
background shell inside the session that submitted them.

Two things are needed and this covers the second: the first is that Slurm itself
records a bad ending (`workflows/on-job-failure.sbatch`), and the second is that a
later session can ask "what did I submit, and how did it end" without having to
remember. Job ids come from a submission record on disk, so the question survives
the session that asked it.

TIMEOUT is not a failure here. A segment cut by the wall clock is the ordinary end of
a chained run and the next segment continues from its checkpoint; the states worth
acting on are FAILED, NODE_FAIL, CANCELLED and OUT_OF_MEMORY.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess

ACTIONABLE = {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}
BENIGN = {"COMPLETED", "RUNNING", "PENDING", "COMPLETING", "TIMEOUT"}


def sacct(job_ids):
    """State, name and elapsed for each id, as Slurm reports it."""
    if not job_ids:
        return {}
    out = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
         "-o", "JobID,JobName,State,Elapsed,End"],
        capture_output=True, text=True, timeout=120).stdout
    rows = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid = parts[0].split(".")[0]
        rows.setdefault(jid, {"job_id": jid, "name": parts[1],
                              "state": parts[2].split()[0] if parts[2] else "",
                              "elapsed": parts[3], "end": parts[4]})
    return rows


def read_record(path):
    """Job ids from a submission record: one `job_id<TAB>label` per line."""
    ids = []
    if not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split("\t")[0])
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="append", default=[],
                    help="a submission record; repeat for several")
    ap.add_argument("--job", action="append", default=[], help="a job id; repeat")
    ap.add_argument("--out", help="TSV to write")
    args = ap.parse_args(argv)

    ids = list(args.job)
    for r in args.record:
        ids.extend(read_record(r))
    ids = sorted(set(ids), key=lambda x: (len(x), x))
    if not ids:
        print("  job id が無い。--record か --job を渡す")
        return 2

    rows = sacct(ids)
    missing = [i for i in ids if i not in rows]
    actionable = [r for r in rows.values() if r["state"] in ACTIONABLE]
    unknown = [r for r in rows.values() if r["state"] not in BENIGN | ACTIONABLE]

    print(f"  照会 {len(ids)} 件 / Slurm が返した {len(rows)} 件")
    for label, group in (("★対処が要る", actionable), ("状態が分からない", unknown)):
        if group:
            print(f"\n  {label}: {len(group)} 件")
            for r in sorted(group, key=lambda x: x["job_id"]):
                print(f"    {r['job_id']:>10s} {r['name'][:38]:38s} {r['state']:14s} "
                      f"{r['elapsed']:>11s} {r['end']}")
    if missing:
        print(f"\n  Slurm に記録が無い（古すぎる）: {len(missing)} 件 {missing[:8]}")
    if not actionable and not unknown:
        print("\n  対処が要るものは無い")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["job_id", "name", "state", "elapsed", "end"],
                               delimiter="\t")
            w.writeheader()
            w.writerows(sorted(rows.values(), key=lambda r: r["job_id"]))
        print(f"\n  {args.out}")
    return 1 if actionable else 0


if __name__ == "__main__":
    raise SystemExit(main())
