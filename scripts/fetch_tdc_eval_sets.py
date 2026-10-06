"""Fetch the downstream evaluation sets from TDC and record what came down.

The machine goes behind a reservation that no user job can run in, so anything not
fetched beforehand cannot be fetched later. This takes the sets the evaluation needs
and writes their row counts and the version that produced them, because a count read
off the corpus months later cannot establish which release it came from.

hERG appears in TDC under more than one name with very different sizes. Splitting a
small set three ways by scaffold leaves each split small enough that the spread
between splits says more about the split than the model, so every candidate is listed
with its count and the largest is marked, rather than one being picked up front.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from importlib.metadata import version

# Solubility is AqSolDB, a regression task. The hERG candidates are the names TDC
# publishes for hERG blockade; which of them is usable is decided by the counts.
TARGETS = [
    ("Tox", "hERG"),
    ("Tox", "hERG_Karim"),
    ("HTS", "hERG_inhib"),
]
SOLUBILITY = ("ADME", "Solubility_AqSolDB")


def load(group: str, name: str, path: str):
    """Pull one dataset. Returns its row count and columns, or the failure."""
    from tdc import single_pred

    cls = getattr(single_pred, group)
    data = cls(name=name, path=path)
    df = data.get_data()
    return {"rows": int(len(df)), "columns": list(df.columns)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, help="directory TDC caches into")
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)

    os.makedirs(args.path, exist_ok=True)
    out = {"pytdc_version": version("PyTDC"), "path": os.path.abspath(args.path),
           "solubility": {}, "herg_candidates": {}}

    group, name = SOLUBILITY
    print(f"=== 水溶解度 {group}/{name} ===", flush=True)
    try:
        got = load(group, name, args.path)
        out["solubility"] = {"group": group, "name": name, **got}
        print(f"  {got['rows']:,} 行  列 {got['columns']}", flush=True)
    except Exception as exc:                        # noqa: BLE001 - recorded, not raised
        out["solubility"] = {"group": group, "name": name, "error": repr(exc)}
        print(f"  取得できない: {exc}", flush=True)
        traceback.print_exc()

    print("\n=== hERG の候補 ===", flush=True)
    for group, name in TARGETS:
        try:
            got = load(group, name, args.path)
            out["herg_candidates"][name] = {"group": group, **got}
            print(f"  {name:14s} {got['rows']:,} 行  列 {got['columns']}", flush=True)
        except Exception as exc:                    # noqa: BLE001 - recorded, not raised
            out["herg_candidates"][name] = {"group": group, "error": repr(exc)}
            print(f"  {name:14s} 取得できない: {exc}", flush=True)

    usable = {k: v["rows"] for k, v in out["herg_candidates"].items() if "rows" in v}
    if usable:
        largest = max(usable, key=usable.get)
        out["herg_largest"] = {"name": largest, "rows": usable[largest]}
        print(f"\n件数が最も多いのは {largest}（{usable[largest]:,} 行）", flush=True)
    else:
        out["herg_largest"] = None
        print("\nhERG はどの候補も取得できなかった", flush=True)

    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n記録 {args.report}  PyTDC {out['pytdc_version']}")
    return 0 if usable and "rows" in out["solubility"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
