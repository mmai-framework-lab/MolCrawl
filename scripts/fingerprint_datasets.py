"""Fingerprint the training_ready_* datasets so they can be checked for tampering.

Three of the five modalities were built before #134 put a seed on the split, so
re-running prepare_gpt2.py does not reproduce them. The copy on disk is the only
original there is: if it is deleted or silently altered, every result measured on
it becomes unverifiable. This records enough to notice that happening.

Full-content hashing is not affordable -- 67 directories, 8.0 TiB, 16,489 data
files. Instead each .arrow/.bin gets a sampled MD5 over its first, middle, and
last 1 MiB, alongside its exact byte size. Size alone catches truncation; the
three windows catch a rewrite that preserves length. A crafted change that keeps
the size and all three windows would slip through, which is not the threat here.

Reads are issued from a thread pool because the cost is Lustre round-trip
latency, not bandwidth (the same finding that drove rna_pack_to_bin.py): ~50k
seeks serialised at ~11 ms each would take hours, and 32 in flight does not.

Output is JSONL, one line per directory, flushed as each finishes -- an
interrupted run still leaves a usable partial ledger, and a later run can skip
what is already there.

Run it on a compute node; the login node is not the place for 48 GiB of reads::

    sbatch workflows/fingerprint-datasets.sbatch
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
WINDOW = 1 << 20
PATTERNS = ("learning_source*/*/training_ready_*", "learning_source*/*/*/training_ready_*")


def sampled_md5(path):
    """MD5 over the first, middle, and last WINDOW bytes."""
    size = os.path.getsize(path)
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for offset in (0, max(0, size // 2 - WINDOW // 2), max(0, size - WINDOW)):
            fh.seek(offset)
            h.update(fh.read(WINDOW))
    return h.hexdigest()


def walk(directory):
    files, total, mtimes, data = 0, 0, [], []
    for root, _, names in os.walk(directory):
        for name in names:
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            files += 1
            total += st.st_size
            mtimes.append(st.st_mtime)
            if name.endswith((".arrow", ".bin")) and not name.startswith("cache-"):
                data.append((os.path.relpath(path, directory), path, st.st_size, st.st_mtime))
    return files, total, mtimes, sorted(data)


def fingerprint(directory, pool):
    files, total, mtimes, data = walk(directory)
    if not files:
        return None
    digests = list(pool.map(lambda d: sampled_md5(d[1]), data))
    members = [
        {
            "file": rel,
            "bytes": size,
            "mtime": datetime.fromtimestamp(mtime, JST).strftime("%Y-%m-%d %H:%M:%S"),
            "sample_md5": digest,
        }
        for (rel, _, size, mtime), digest in zip(data, digests)
    ]
    joined = "|".join(f"{m['file']}:{m['bytes']}:{m['sample_md5']}" for m in members)
    return {
        "dir": directory,
        "files": files,
        "bytes": total,
        "n_data_files": len(members),
        "mtime_min": datetime.fromtimestamp(min(mtimes), JST).strftime("%Y-%m-%d %H:%M"),
        "mtime_max": datetime.fromtimestamp(max(mtimes), JST).strftime("%Y-%m-%d %H:%M"),
        "fingerprint": hashlib.md5(joined.encode()).hexdigest(),
        "members": members,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="directory the glob patterns are relative to")
    ap.add_argument("--out", required=True, help="JSONL ledger; existing directories in it are skipped")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    os.chdir(args.root)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["dir"])
                except (ValueError, KeyError):
                    pass
        print(f"resuming: {len(done)} directories already in {args.out}", flush=True)

    dirs = sorted({d for p in PATTERNS for d in glob.glob(p) if os.path.isdir(d)})
    todo = [d for d in dirs if d not in done]
    print(f"{len(dirs)} directories, {len(todo)} to do", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(args.out, "a") as out:
        for i, directory in enumerate(todo, 1):
            t0 = time.time()
            row = fingerprint(directory, pool)
            if row is None:
                print(f"[{i}/{len(todo)}] empty, skipped: {directory}", flush=True)
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{i}/{len(todo)}] {time.time() - t0:7.1f}s "
                f"{row['bytes'] / 2**30:8.2f} GiB {row['n_data_files']:>5} files  "
                f"{row['fingerprint'][:16]}  {directory}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
