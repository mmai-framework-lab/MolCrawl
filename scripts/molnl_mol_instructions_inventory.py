#!/usr/bin/env python3
"""Everything the Mol-Instructions retrieval measurements left behind, counted by machine.

The 2026-11-10 judgment approves deleting these in 3.3 and asks that the figures be
counted again mechanically before anything goes. A hand-picked candidate set has twice
now answered a completeness question wrongly here -- the first count of the measurement
files was 64 files and 52,035 rows against the true 96 and 52,003 -- so the scope is
decided by stated rules and a scan, and the deletion reads the manifest the scan wrote
rather than a list anyone typed.

The rules, in the order they are applied:

  measurements  a directory under the report root holding ``retrieval-*.json`` whose
                summaries do not name the frozen SMolInstruct test pairs is a
                Mol-Instructions measurement directory. Every file under it is in scope.
                A summary from the test-split round carries ``pairs_file``; the
                Mol-Instructions round predates that field.
  figures       a figure, its data table or its caption is in scope when it sits under a
                molnl figure directory and either its name carries ``retrieval`` or a
                file of the same stem is itself in scope. A table anywhere under the
                report root is in scope when it names zjunlp/Mol-Instructions as the
                evaluation set it was measured on.
  tables        a table at the report root is in scope when its rows carry a frame of a
                measurement directory in scope -- which is how the 250-query
                text-to-molecule round is caught, since it names no dataset in itself.

Reports are never in scope. A .md or .html file says what was measured and why, and the
judgment asks for the measurements to go, not for the record of them.

    python scripts/molnl_mol_instructions_inventory.py --report-root <dir> \
        --manifest <json>                        # count and write the manifest
    python scripts/molnl_mol_instructions_inventory.py --manifest <json> --delete
"""

from __future__ import annotations

import argparse
import glob
import json
import os

KEEP_SUFFIXES = (".md", ".html", ".ja.md")
TEXT_SUFFIXES = (".tsv", ".txt", ".json", ".csv")
MOL_INSTRUCTIONS = "zjunlp/Mol-Instructions"


def is_smoltest_summary(path):
    """Does this retrieval summary belong to the SMolInstruct test-split round?"""
    try:
        summary = json.load(open(path, encoding="utf-8"))
    except Exception:
        return False
    pairs = summary.get("pairs_file") or ""
    return "smoltest" in pairs or "smolinstruct-test" in os.path.basename(pairs).lower()


def measurement_dirs(root):
    """Directories of retrieval summaries that are not the test-split round."""
    found = {}
    for path in glob.glob(os.path.join(root, "**", "retrieval-*.json"), recursive=True):
        directory = os.path.dirname(path)
        found.setdefault(directory, []).append(path)
    out = {}
    for directory, summaries in sorted(found.items()):
        smoltest = sum(1 for p in summaries if is_smoltest_summary(p))
        out[directory] = {"summaries": len(summaries), "smoltest_summaries": smoltest,
                          "in_scope": smoltest == 0}
    return out


def rank_rows(paths):
    """Data rows of the per-query rank tables, with the rt1- copies counted apart."""
    original = duplicate = 0
    for path in paths:
        if not path.endswith("-ranks.tsv"):
            continue
        with open(path, encoding="utf-8") as fh:
            rows = sum(1 for line in fh
                       if line.strip() and not line.startswith("#")
                       and not line.startswith("model\t"))
        if os.path.basename(path).startswith("rt1-"):
            duplicate += rows
        else:
            original += rows
    return original, duplicate


def names_mol_instructions(path):
    if not path.endswith(TEXT_SUFFIXES) or os.path.getsize(path) > 50_000_000:
        return False
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return MOL_INSTRUCTIONS in fh.read()
    except OSError:
        return False


def frames_of(paths):
    """The model labels a measurement directory scored, to catch tables about them."""
    labels = set()
    for path in paths:
        if not path.endswith(".json"):
            continue
        try:
            summary = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if summary.get("label"):
            labels.add(str(summary["label"]))
    return labels


def inventory(root):
    dirs = measurement_dirs(root)
    scope_dirs = [d for d, info in dirs.items() if info["in_scope"]]
    measurements = []
    for directory in scope_dirs:
        for base, _sub, files in os.walk(directory):
            measurements.extend(os.path.join(base, f) for f in sorted(files))
    measurements.sort()
    labels = frames_of(measurements)

    figures, tables = [], []
    for base, sub, files in os.walk(root):
        sub[:] = [d for d in sub if os.path.join(base, d) not in scope_dirs]
        if any(os.path.join(base, "").startswith(os.path.join(d, ""))
               for d in scope_dirs):
            continue
        in_figure_dir = os.path.basename(base).startswith("molnl-figs-")
        for name in sorted(files):
            path = os.path.join(base, name)
            if name.endswith(KEEP_SUFFIXES) and not name.endswith(".caption.txt"):
                continue
            if in_figure_dir and "retrieval" in name.lower():
                figures.append(path)
                continue
            if names_mol_instructions(path):
                (figures if in_figure_dir else tables).append(path)
                continue
            # A table of the measurements that names no dataset of its own: caught by
            # the model labels the measurement directories scored.
            if name.endswith(".tsv") and "retrieval" in name.lower() and labels:
                try:
                    head = open(path, encoding="utf-8", errors="ignore").read(200_000)
                except OSError:
                    continue
                if any(f"\n{label}\t" in head for label in labels):
                    tables.append(path)
    # A figure beside a data table already in scope.
    stems = {os.path.splitext(p)[0] for p in figures}
    for base, _sub, files in os.walk(root):
        if not os.path.basename(base).startswith("molnl-figs-"):
            continue
        for name in sorted(files):
            path = os.path.join(base, name)
            if path in figures:
                continue
            if os.path.splitext(path)[0] in stems:
                figures.append(path)
    figures.sort()
    tables.sort()

    original, duplicate = rank_rows(measurements)
    groups = {
        "measurements": measurements,
        "figures": figures,
        "tables": tables,
    }
    def size(paths):
        return sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    return {
        "report_root": os.path.abspath(root),
        "retrieval_directories": {d: info for d, info in dirs.items()},
        "measurement_directories": scope_dirs,
        "rank_rows_original": original,
        "rank_rows_in_rt1_copies": duplicate,
        "counts": {k: len(v) for k, v in groups.items()},
        "bytes": {k: size(v) for k, v in groups.items()},
        "png_in_figures": sum(1 for p in figures if p.endswith(".png")),
        "groups": groups,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-root")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--delete", action="store_true",
                    help="remove exactly the paths the manifest lists, nothing else")
    a = ap.parse_args()

    if not a.delete:
        if not a.report_root:
            raise SystemExit("--report-root is required unless --delete is given")
        found = inventory(a.report_root)
        for directory, info in found["retrieval_directories"].items():
            mark = "DELETE" if info["in_scope"] else "keep  "
            print(f"  {mark} {directory}: {info['summaries']} summaries, "
                  f"{info['smoltest_summaries']} of the test-split round", flush=True)
        for group, paths in found["groups"].items():
            print(f"  {group}: {len(paths)} files, "
                  f"{found['bytes'][group] / 1e6:.1f} MB", flush=True)
            for path in paths:
                print(f"      {path}")
        print(f"  rank rows: {found['rank_rows_original']:,} original, "
              f"{found['rank_rows_in_rt1_copies']:,} more in the rt1- copies")
        print(f"  figures that are .png: {found['png_in_figures']}")
        os.makedirs(os.path.dirname(os.path.abspath(a.manifest)), exist_ok=True)
        json.dump(found, open(a.manifest, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {a.manifest}")
        return 0

    found = json.load(open(a.manifest, encoding="utf-8"))
    removed = missing = 0
    for group, paths in found["groups"].items():
        for path in paths:
            if not os.path.exists(path):
                missing += 1
                continue
            os.remove(path)
            removed += 1
            print(f"  removed {path}")
    for directory in sorted(found["measurement_directories"], key=len, reverse=True):
        for base, dirs_, _files in os.walk(directory, topdown=False):
            for name in dirs_:
                try:
                    os.rmdir(os.path.join(base, name))
                except OSError:
                    pass
        try:
            os.rmdir(directory)
            print(f"  removed directory {directory}")
        except OSError as exc:
            print(f"  kept directory {directory}: {exc}")
    print(f"  removed {removed} files, {missing} were already gone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
