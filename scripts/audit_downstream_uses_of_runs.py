#!/usr/bin/env python3
"""Which downstream-evaluation outputs name a given set of training runs.

Directive `all-checkpoint-metric-order-2026-10-28-v2.md` §5: for the runs where the
saved best checkpoint is not the one the judged metric picked, list the downstream
evaluations that used them. Nothing is re-run; this only reports what mentions what.

A run is looked for by its directory name and by its full output path, because the
downstream results record it both ways: the probe tables keep the run name in a
column, the ProteinGym and ClinVar outputs keep the checkpoint path.

Only text-shaped files are opened (by extension), and each is read once, so this is
a scan over result files rather than over the checkpoints themselves.

Usage:
    python scripts/audit_downstream_uses_of_runs.py \
        --audit <checkpoint-metric-match.json> --match no \
        --search <dir> [--search <dir> ...] --report <out.json>
"""

from __future__ import annotations

import argparse
import json
import os
import re

# A run name is a prefix of other run names -- `bert-small-global_random_seed1` of
# `bert-small-global_random_seed1-w1026`, `compounds-medium-lr2p8e4` of
# `compounds-medium-lr2p8e4-seed42` -- and so is its output path. A plain substring
# search therefore reports the sibling run's evaluation as this run's: it credited
# 21 genome runs and one compounds run with evaluations that read a different
# checkpoint. A hit has to end at a character that cannot continue the name.
NAME_TAIL = r"(?![A-Za-z0-9_.\-])"

TEXT_SUFFIXES = (".json", ".tsv", ".csv", ".md", ".txt", ".log", ".out", ".yaml",
                 ".yml", ".sh", ".html")
# A result file large enough to be a data dump rather than a report is still read,
# but one this big is a checkpoint shard or an archive that slipped past the suffix
# filter and is skipped rather than pulled into memory.
MAX_BYTES = 64 * 1024 * 1024


def flagged_runs(audit_path, match):
    with open(audit_path) as fh:
        audit = json.load(fh)
    runs = [r for r in audit["runs"] if r.get("match") == match]
    needles = {}
    for r in runs:
        key = f"{r['modality']}/{r['arch']}/{r['run']}"
        # The basename alone is ambiguous across trees -- `compounds-small-lr1e3`
        # exists in more than one -- so the path is searched for as well and the
        # basename hit is reported as the weaker of the two.
        needles[key] = {
            "name": r["run"],
            "path": r["run_dir"],
            # The path may be followed by a separator -- it is a directory that a
            # checkpoint hangs off -- but not by more name.
            "path_re": re.compile(re.escape(r["run_dir"]) + NAME_TAIL),
            "name_re": re.compile(re.escape(r["run"]) + NAME_TAIL),
            "row": r,
        }
    return needles


def scan(roots, needles):
    hits = {key: {"by_path": [], "by_name": []} for key in needles}
    files_read = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # Never descend into a checkpoint directory: it holds weights, not results.
            dirnames[:] = [d for d in dirnames if not d.startswith("checkpoint-")]
            for name in filenames:
                if not name.endswith(TEXT_SUFFIXES):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(path) > MAX_BYTES:
                        continue
                    with open(path, errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    continue
                files_read += 1
                for key, needle in needles.items():
                    if needle["path_re"].search(text):
                        hits[key]["by_path"].append(path)
                    elif needle["name_re"].search(text):
                        hits[key]["by_name"].append(path)
    return hits, files_read


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--match", default="no",
                    help="which runs to look for: the value of the audit's `match` field")
    ap.add_argument("--search", action="append", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)

    needles = flagged_runs(args.audit, args.match)
    print(f"{len(needles)} runs with match={args.match!r}", flush=True)
    roots = []
    for root in args.search:
        if not os.path.isdir(root):
            raise SystemExit(f"not a directory: {root}")
        roots.append(root)
        print(f"  search: {root}", flush=True)

    hits, files_read = scan(roots, needles)
    print(f"read {files_read:,} text files", flush=True)

    used, unused = {}, []
    for key, hit in hits.items():
        if hit["by_path"] or hit["by_name"]:
            used[key] = hit
        else:
            unused.append(key)

    print(f"\n{len(used)} of {len(needles)} runs are named by something; "
          f"{len(unused)} are named by nothing", flush=True)
    for key in sorted(used):
        hit = used[key]
        print(f"  {key}", flush=True)
        for path in sorted(set(hit["by_path"]))[:8]:
            print(f"      path: {path}", flush=True)
        for path in sorted(set(hit["by_name"]))[:8]:
            print(f"      name: {path}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump({
            "match": args.match,
            "runs_searched": sorted(needles),
            "files_read": files_read,
            "used_by": {k: {"by_path": sorted(set(v["by_path"])),
                            "by_name": sorted(set(v["by_name"]))}
                        for k, v in used.items()},
            "named_by_nothing": sorted(unused),
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
