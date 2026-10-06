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

# Solubility is AqSolDB, a regression task. The hERG candidates are every name TDC
# publishes for hERG under Tox -- retrieve_dataset_names("Tox") lists exactly these
# three and no other group has any -- and which one is used is decided by the counts.
# herg_central carries three labels rather than one, so it needs a label named.
# hERG_inhib is the binary one; hERG_at_1uM and hERG_at_10uM are percent inhibition
# and would make the largest set the only regression among the three, which is not a
# comparison. hERG_inhib is also not a dataset of its own, which is easy to read it as.
TARGETS = [
    ("Tox", "hERG", None),
    ("Tox", "hERG_Karim", None),
    ("Tox", "herg_central", "hERG_inhib"),
]
SOLUBILITY = ("ADME", "Solubility_AqSolDB")


def load(group: str, name: str, path: str, label: str | None = None):
    """Pull one dataset. Returns its row count and columns, or the failure."""
    from tdc import single_pred

    cls = getattr(single_pred, group)
    data = cls(name=name, path=path) if label is None \
        else cls(name=name, path=path, label_name=label)
    df = data.get_data()
    return {"rows": int(len(df)), "columns": list(df.columns),
            **({"label": label} if label else {})}


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
    for group, name, label in TARGETS:
        try:
            got = load(group, name, args.path, label)
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
