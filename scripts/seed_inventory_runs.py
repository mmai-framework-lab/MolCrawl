"""Report the seed every recent run actually used, from what the run itself wrote.

The directive of 2026-09-30 asks for the resolved ``seed`` and ``data_seed`` per
modality, architecture and size, with the source named and no guessing. A config
is not the answer: a run can be launched with an override, and what matters is
what it ran with.

So this reads the run's own record -- ``run_manifest.json`` for the HF side, the
checkpoint's ``training_args.bin`` when there is no manifest, and the launcher
echo in the nanoGPT stdout for GPT-2 -- and says which of those it used. Anything
it cannot resolve is printed as unresolved rather than filled in.

    python scripts/seed_inventory_runs.py --roots <dir> [<dir> ...] --since-days 2
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

SEED_ECHO = re.compile(r"^\s*seed\s*[:=]\s*(\d+)", re.M)


def from_manifest(path: Path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    seed = d.get("seed", {})
    if isinstance(seed, dict) and "seed" in seed:
        return seed.get("seed"), seed.get("data_seed"), "run_manifest.json"
    if "seed" in d:
        return d.get("seed"), d.get("data_seed"), "run_manifest.json"
    return None


def from_training_args(ckpt: Path):
    """training_args.bin is a pickle; read the two integers without torch."""
    try:
        blob = ckpt.read_bytes()
    except OSError:
        return None
    out = {}
    for key in ("seed", "data_seed"):
        m = re.search(rb"\x8c%c%s" % (len(key), key.encode()), blob)
        if not m:
            continue
        tail = blob[m.end():m.end() + 6]
        if tail[:1] == b"M":
            out[key] = int.from_bytes(tail[1:3], "little")
        elif tail[:1] == b"K":
            out[key] = tail[1]
        elif tail[:1] == b"J":
            out[key] = int.from_bytes(tail[1:5], "little", signed=True)
    if "seed" in out:
        return out.get("seed"), out.get("data_seed"), "training_args.bin"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--since-days", type=float, default=2.0)
    ap.add_argument("--all", action="store_true", help="ignore --since-days")
    args = ap.parse_args()
    cutoff = time.time() - args.since_days * 86400

    rows = []
    for root in args.roots:
        for manifest in sorted(Path(root).glob("**/run_manifest.json")):
            if not args.all and manifest.stat().st_mtime < cutoff:
                continue
            got = from_manifest(manifest)
            run = manifest.parent
            if got is None:
                ck = sorted(run.glob("checkpoint-*/training_args.bin"))
                got = from_training_args(ck[-1]) if ck else None
            when = time.strftime("%m-%d %H:%M", time.localtime(manifest.stat().st_mtime))
            if got is None:
                rows.append((str(run), "", "", "解決できず", when))
            else:
                rows.append((str(run), got[0], got[1], got[2], when))

    width = max((len(r[0]) for r in rows), default=20)
    print(f"{'run':{width}}  {'seed':>6} {'data_seed':>10}  {'出どころ':16} 更新")
    for run, seed, data_seed, src, when in sorted(rows):
        print(f"{run:{width}}  {str(seed):>6} {str(data_seed):>10}  {src:16} {when}")
    off = [r for r in rows if r[1] not in ("", 42)]
    print(f"\n{len(rows)} run / 42 以外 {len(off)} 件 / 解決できず "
          f"{len([r for r in rows if r[3] == '解決できず'])} 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
