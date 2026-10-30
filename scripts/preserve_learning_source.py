"""Hash every file under the learning_source* trees, so they can be verified later.

genome, protein and rna were built before #134 seeded the split, so re-running
prepare_gpt2.py does not reproduce them: the copy on disk is the only original
there is. The checkpoints are worse off -- neither nanoGPT's config nor
HuggingFace's TrainingArguments records which dataset directory a run read, so
a run cannot even say what it trained on. Nothing here can be regenerated.

So this hashes all of it, with no exceptions -- duplicates included. Five groups
of directories are known to be byte-identical (rna at 328 GiB, molecule_nat_lang
twice, compounds' gpt2/bert pairs), and they are hashed anyway: which copy is
the redundant one is not this script's call to make.

Full MD5, not the sampled fingerprint of fingerprint_datasets.py. That one reads
three 1 MiB windows to answer "has this been altered" cheaply for the 67
training_ready_* directories; this answers it exactly, for everything. Size and
mtime are recorded alongside, so an interrupted run still leaves the weaker
ledger over whatever it reached.

Directories are hashed concurrently, not the files within one directory. That
distinction is the whole throughput story: a single Lustre stream reads at about
150 MiB/s, so the rate is set by how many reads are in flight, and a checkpoint
directory holds four files. Hashing one directory at a time capped the job at
four streams -- 611 MiB/s whether the pool had 16 workers or 48. Running one
directory per worker keeps all of them busy across the 2,906 directories.

Output is JSONL, one line per directory, flushed as each finishes. A re-run
skips directories already in the file, so the job can be resumed after a
timeout and loses at most one directory in flight per worker. Lines are written
as each directory completes rather than in submission order, so one slow
directory does not hold the ledger back.

Run it on a compute node::

    sbatch workflows/preserve-learning-source.sbatch
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
CHUNK = 8 << 20


def full_md5(path):
    h = hashlib.md5()
    with open(path, "rb", buffering=0) as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def leaf_dirs(roots):
    """Every directory that directly contains at least one file."""
    found = []
    for root in roots:
        for directory, _, names in os.walk(root):
            if names:
                found.append(directory)
    return sorted(found)


def stamp(mtime):
    return datetime.fromtimestamp(mtime, JST).strftime("%Y-%m-%d %H:%M:%S")


def do_dir(directory):
    entries = []
    with os.scandir(directory) as it:
        for entry in it:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            entries.append((entry.name, entry.path, st.st_size, st.st_mtime))
    entries.sort()
    if not entries:
        return None

    digests = [_safe(path) for _, path, _, _ in entries]
    failures = [name for (name, *_), digest in zip(entries, digests) if digest is None]

    members = [
        {"file": name, "bytes": size, "mtime": stamp(mtime), "md5": digest}
        for (name, _, size, mtime), digest in zip(entries, digests)
    ]
    joined = "|".join(f"{m['file']}:{m['bytes']}:{m['md5']}" for m in members)
    return {
        "dir": directory,
        "files": len(members),
        "bytes": sum(m["bytes"] for m in members),
        "unreadable": failures,
        "fingerprint": hashlib.md5(joined.encode()).hexdigest(),
        "members": members,
    }


def directory_label(directory, width=88):
    return directory if len(directory) <= width else "..." + directory[-(width - 3) :]


def _safe(path):
    try:
        return full_md5(path)
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="directory the glob is relative to")
    ap.add_argument("--glob", default="learning_source*", help="trees to cover")
    ap.add_argument("--out", required=True, help="JSONL ledger; directories already in it are skipped")
    ap.add_argument("--workers", type=int, default=16)
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
        print(f"resuming: {len(done)} directories already recorded", flush=True)

    roots = sorted(d for d in glob.glob(args.glob) if os.path.isdir(d))
    print(f"trees: {len(roots)} -- {', '.join(roots)}", flush=True)

    t0 = time.time()
    dirs = leaf_dirs(roots)
    print(f"walked in {time.time() - t0:.0f}s: {len(dirs)} directories hold files", flush=True)

    todo = [d for d in dirs if d not in done]
    print(f"{len(todo)} to hash", flush=True)

    hashed_bytes, started, i = 0, time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(args.out, "a") as out:
        futures = [pool.submit(do_dir, d) for d in todo]
        for future in as_completed(futures):
            row = future.result()
            i += 1
            if row is None:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            hashed_bytes += row["bytes"]
            elapsed = max(time.time() - started, 1e-9)
            bad = f" UNREADABLE={len(row['unreadable'])}" if row["unreadable"] else ""
            print(
                f"[{i}/{len(todo)}] {row['bytes'] / 2**30:8.2f} GiB "
                f"{row['files']:>5} files  cumulative {hashed_bytes / 2**40:6.3f} TiB "
                f"at {hashed_bytes / 2**20 / elapsed:6.0f} MiB/s  {row['fingerprint'][:16]}  "
                f"{directory_label(row['dir'])}{bad}",
                flush=True,
            )
    print(f"done: {hashed_bytes / 2**40:.3f} TiB in {(time.time() - started) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
