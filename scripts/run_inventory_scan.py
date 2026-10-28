"""Walk the run trees and write one row per training run.

The inventory published in 2026-08-31 had to recover its numbers from ``ckpt.pt``
and ``training_args.bin``, which are binary and carry no dataset path. Runs made
since then write ``run_manifest.json`` beside their checkpoints, in plain text,
and it records what the binaries could not: which dataset was read, the resolved
learning rate and seed, the effective global batch, and the commit that ran.

So this script prefers the manifest and falls back to what the directory shows.
It reads no checkpoint: steps come from the ``checkpoint-N`` directory names and
the times from file names and mtimes, so a tree of several TB is walked in
seconds rather than parsed.

Paths live in the config file, not here: this repository is public.

    python scripts/run_inventory_scan.py --config <file.json> --out <file.tsv>

The config names the trees:

    {"roots": [{"tree": "learning_source_protein_runs", "path": "/.../..."}]}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

TFEVENTS_RE = re.compile(r"events\.out\.tfevents\.(\d{9,11})\.")
LOGGING_RE = re.compile(r"logging_(\d{8})_(\d{6})\.csv$")
CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

COLUMNS = ["tree", "run", "host", "origin", "start", "end", "hours", "steps", "arch",
           "batch", "lr", "seed", "max_iters", "dataset", "params_rec", "curve_rec"]


def stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def is_run(d: Path) -> bool:
    """A directory holding a run, rather than one holding runs.

    Four shapes, because four generations of this project wrote four layouts:
    a manifest, HF checkpoints, a nanoGPT checkpoint, or only its logs.
    """
    if (d / "run_manifest.json").exists() or (d / "ckpt.pt").exists():
        return True
    for child in d.iterdir():
        if child.is_dir() and CKPT_RE.match(child.name):
            return True
        if child.is_file() and (LOGGING_RE.search(child.name) or TFEVENTS_RE.search(child.name)):
            return True
    return False


def started_at(d: Path, files):
    """When the run began, from the names its own tooling wrote into the tree.

    mtimes are useless for this: the runs carried over from the old machine all
    share the day they were copied. A file name does not move when it is copied.
    """
    best = None
    for f in files:
        m = TFEVENTS_RE.search(f.name)
        if m:
            best = min(best, int(m.group(1))) if best else int(m.group(1))
            continue
        m = LOGGING_RE.search(f.name)
        if m:
            t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
            best = min(best, t) if best else t
    return best


def last_logged_iter(files):
    """The largest iteration any of a nanoGPT run's logs reached."""
    best = None
    for f in files:
        if not LOGGING_RE.search(f.name):
            continue
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                rows = fh.readlines()
        except OSError:
            continue
        for line in reversed(rows):
            head = line.split(",")[0].strip()
            if head.isdigit():
                best = max(best, int(head)) if best is not None else int(head)
                break
    return best


def read_manifest(d: Path):
    try:
        return json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def dig(obj, *path, default=""):
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return default if obj is None else obj


def relative(path: str, strip: str) -> str:
    """The dataset as a path inside the project, not a bare directory name.

    "training_ready_hf_dataset" is the name of four different datasets; what makes
    it answerable is the path it sits at.

    A run staged to node-local NVMe records the copy it read, under /tmp/<job>/.
    The tree it was copied from is not in the manifest, so what is kept is the
    part that identifies the dataset -- modality, subset, build -- and the "/tmp"
    prefix is dropped rather than printed as if it were where the data lives.
    """
    if path.startswith("/tmp/"):
        parts = path.split("/", 3)
        return parts[3] if len(parts) > 3 else path
    return path[len(strip):] if strip and path.startswith(strip) else path


def scan_run(tree: str, root: Path, d: Path, strip: str = ""):
    files = [f for f in d.iterdir() if f.is_file()]
    ckpts = [(int(m.group(1)), c) for c in d.iterdir()
             if c.is_dir() and (m := CKPT_RE.match(c.name))]

    row = {c: "" for c in COLUMNS}
    row["tree"] = tree
    row["run"] = str(d.relative_to(root))

    # Both write "checkpoint-N" directories, so that does not tell them apart:
    # what does is trainer_state.json inside one of them, against nanoGPT's single
    # ckpt.pt at the top. Reading the number off the directory name cost 226 runs
    # their architecture until this was checked against the published table.
    hf = (d / "trainer_state.json").exists() or any(
        (c / "trainer_state.json").exists() for _, c in ckpts)
    row["arch"] = "HF" if hf else "nanoGPT"

    if hf and ckpts:
        row["steps"] = max(n for n, _ in ckpts)
    elif not hf:
        # nanoGPT checkpoints every few thousand steps, so the last one understates
        # where the run actually stopped. Its logs carry every evaluation, and the
        # run's segments each wrote their own file, so the last step is the largest
        # of their last rows -- not the last file's.
        last = last_logged_iter(files)
        row["steps"] = last if last is not None else (
            max(n for n, _ in ckpts) if ckpts else "")

    mtimes = [f.stat().st_mtime for f in files] + [c.stat().st_mtime for _, c in ckpts]
    start = started_at(d, files)
    end = max(mtimes) if mtimes else None

    man = read_manifest(d)
    if man:
        written = dig(man, "written")
        if written and not start:
            try:
                start = datetime.fromisoformat(written).timestamp()
            except ValueError:
                pass
        row["host"] = dig(man, "run", "node") or "—"
        row["batch"] = dig(man, "batch", "effective_global_batch")
        row["lr"] = dig(man, "schedule", "learning_rate")
        row["seed"] = dig(man, "seed", "seed")
        row["max_iters"] = dig(man, "schedule", "max_steps")
        ds = dig(man, "data", "dataset_dir")
        row["dataset"] = relative(ds, strip) if ds else ""
        row["params_rec"] = "run_manifest"
    else:
        row["host"] = "—"
        row["params_rec"] = ("trainer_state" if (d / "trainer_state.json").exists()
                             or any((c / "trainer_state.json").exists() for _, c in ckpts)
                             else "ckpt.pt(config)" if (d / "ckpt.pt").exists() else "なし")

    row["start"] = stamp(start) if start else "—"
    row["end"] = stamp(end) if end else "—"
    row["hours"] = f"{(end - start) / 3600:.1f}" if start and end and end > start else "不明"
    # A manifest is written by the run itself on this cluster, so it settles the
    # question. Timestamps do not: the runs carried over from the old machine have
    # an mtime of the day they were copied, which is later than their start and
    # would read as if they had run here.
    row["origin"] = "実走行" if man else "不明"

    kinds = []
    if any(CKPT_RE.match(c.name) and (c / "trainer_state.json").exists() for _, c in ckpts):
        kinds.append("trainer_state")
    if any(LOGGING_RE.search(f.name) for f in files):
        kinds.append("logging_csv")
    if any(TFEVENTS_RE.search(f.name) for f in files):
        kinds.append("tfevents")
    row["curve_rec"] = "+".join(kinds) if kinds else "なし"
    return row


def walk(tree: str, root: Path, rows, seen, strip=""):
    """Depth-first, stopping at a run: a run's checkpoints are not runs."""
    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        print(f"  ! {root}: {exc}", file=sys.stderr)
        return
    stack = list(entries)
    while stack:
        d = stack.pop()
        if CKPT_RE.match(d.name):
            continue
        try:
            if is_run(d):
                key = (tree, str(d.relative_to(root)))
                if key not in seen:
                    seen.add(key)
                    rows.append(scan_run(tree, root, d, strip))
                continue
            stack.extend(p for p in d.iterdir() if p.is_dir())
        except OSError as exc:
            print(f"  ! {d}: {exc}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    rows, seen = [], set()
    for spec in cfg["roots"]:
        root = Path(spec["path"])
        if not root.is_dir():
            print(f"  ! missing: {root}", file=sys.stderr)
            continue
        before = len(rows)
        walk(spec["tree"], root, rows, seen, cfg.get("strip", ""))
        print(f"  {spec['tree']:42s} {len(rows) - before:4d} run", file=sys.stderr)

    rows.sort(key=lambda r: (r["start"], r["tree"], r["run"]))
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} run)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
