#!/usr/bin/env python3
"""One row per BERT run: the seed its config named, and the seed its manifest records.

Directive `all-seed-and-metric-order-2026-11-04b` §3. The two seeds are separate
columns because they disagree on six runs: between ``0f3e7067`` (2026-10-02) and
``97069327`` (2026-10-30) ``main.py`` passed only ``data_seed=`` to
``TrainingArguments``, so ``transformers`` seeded from its own default 42 while the
config named 1 or 17. ``protein_sequence-large-lr5e5-s17`` is one of them: the name
says 17, the manifest says 42.

Which value goes in which column:

* ``config_seed`` -- ``sources.seed.value`` in ``run_manifest.json``: what the config
  asked for.
* ``manifest_seed`` -- ``seed.seed``: the value ``transformers`` actually seeded with,
  read off ``getattr(args, "seed")`` when the run started.

Runs older than ``run_manifest.json`` have neither. Their checkpoints still hold
``training_args.bin``, so ``args_seed`` carries what that file says and the two
directive columns stay empty rather than being filled from a different source.

Run discovery is `audit_checkpoint_metric_match.walk_root`, so this file and the
checkpoint audit cannot disagree about which directories are runs. GPT-2 rows are
dropped here: nanoGPT seeds itself (``train.py`` calls
``torch.manual_seed(seed + ddp_rank)``) and is not what §3 asks about.

Reads ``run_manifest.json`` and, only for the runs without one, the newest
checkpoint's ``training_args.bin``. No weight file is opened and nothing is written
inside a run directory.

Usage:
    python scripts/bert_seed_inventory.py \
        --root compounds=<dir> --root rna=<dir> --out <out.tsv>
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_checkpoint_metric_match import (  # noqa: E402
    CHECKPOINT_RE,
    dig,
    is_production,
    model_size,
    read_manifest,
    walk_root,
)

#: The report TSV convention (docs/07-reports/report_tsv_columns.md) version this
#: file claims in its name and in its first line.
RT_VERSION = 1

#: §3's columns first, in the order the directive lists them, then what is needed to
#: say where each seed came from. ASCII only: convention §6, decision 3.
COLUMNS = [
    "modality",
    "size",
    "lr",
    "run",
    "config_seed",
    "manifest_seed",
    "seed_agrees",
    "args_seed",
    "seed_from",
    "seed_source",
    "production",
    "job_id",
    "steps_run",
    "tree",
    "note",
]


def read_args_seed_from_checkpoint(run_dir):
    """``args.seed`` out of the newest checkpoint's ``training_args.bin``.

    The fallback for runs that predate ``run_manifest.json``. ``training_args.bin``
    is a pickled ``TrainingArguments``, so ``weights_only=False`` is required under
    torch 2.6 and later, where the default flipped. Returns ``(seed, source)``, or
    ``(None, reason)`` when it cannot be read -- a run whose seed cannot be
    established has to say so, not be left looking seedless.
    """
    try:
        entries = sorted(
            (int(CHECKPOINT_RE.match(name).group(1)), name)
            for name in os.listdir(run_dir)
            if CHECKPOINT_RE.match(name)
        )
    except OSError as exc:
        return None, f"run directory unreadable: {exc.__class__.__name__}"
    for _step, name in reversed(entries):
        path = os.path.join(run_dir, name, "training_args.bin")
        if not os.path.isfile(path):
            continue
        try:
            import torch

            args = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - one unreadable file is not fatal
            return None, f"{name}/training_args.bin unreadable: {exc.__class__.__name__}"
        seed = getattr(args, "seed", None)
        if seed is None:
            return None, f"{name}/training_args.bin has no seed"
        return int(seed), f"{name}/training_args.bin"
    return None, "no training_args.bin in any checkpoint"


def tree_label(root):
    """The run tree, named by its last two path components.

    A bare basename is not enough: three of the compounds roots end in
    ``compounds`` (``learning_source_20260805_compounds_packed/compounds`` and two
    others), so the column and the provenance block would say the same word for
    three different trees.
    """
    parts = [p for p in root.rstrip("/").split(os.sep) if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else root)


def produced_something(row):
    """Whether this run left anything a result could be read from.

    ``is_production`` asks whether the run's *name* and job id say it was meant to
    stand behind a result. It cannot see that a run wrote no checkpoint at all: the
    two throughput probes under ``learning_source_bert_tput`` hold a manifest and
    nothing else, and one of their names matches no diagnostic word. A run with no
    checkpoint and no steps produced nothing, whatever it was called.
    """
    if row.get("checkpoints_present"):
        return True
    return bool(row.get("steps_run"))


def inventory_row(row, tree, read_args_seed=read_args_seed_from_checkpoint, note=""):
    """The §3 row for one BERT run.

    ``read_args_seed`` is a parameter so the fallback can be exercised without a
    pickled ``TrainingArguments`` on disk.
    """
    manifest = read_manifest(row["run_dir"]) or {}
    config_seed = dig(manifest, "sources.seed.value")
    manifest_seed = dig(manifest, "seed.seed")
    out = {
        "modality": row["modality"],
        "size": row.get("size") or "",
        "lr": row.get("learning_rate") or "",
        "run": row["run"],
        "note": note,
        "config_seed": "" if config_seed is None else int(config_seed),
        "manifest_seed": "" if manifest_seed is None else int(manifest_seed),
        "seed_agrees": "",
        "args_seed": "",
        "seed_from": dig(manifest, "sources.seed.from") or "",
        "seed_source": "run_manifest.json" if manifest else "",
        "production": "yes" if produced_something(row) and is_production(row) else "no",
        "job_id": row.get("job_id") or "",
        "steps_run": row.get("steps_run") or "",
        "tree": tree,
    }
    if manifest_seed is not None and config_seed is not None:
        out["seed_agrees"] = "yes" if int(manifest_seed) == int(config_seed) else "no"
    if manifest_seed is None:
        seed, source = read_args_seed(row["run_dir"])
        out["args_seed"] = "" if seed is None else int(seed)
        out["seed_source"] = source
    if not produced_something(row):
        out["note"] = "; ".join(x for x in (note, "manifest only (no checkpoint)") if x)
    return out


def row_without_trainer_state(entry, modality):
    """An audit-shaped row for a run `walk_root` could not read a history out of.

    §3 asks for every BERT run, and a run that crashed before writing a
    ``trainer_state.json`` still named a seed. Dropping it would make the inventory
    shorter than the directory listing, which is the one thing a complete list may
    not be. Everything the metric audit takes from ``trainer_state.json`` is absent
    here, so only what the manifest holds is filled in.
    """
    manifest = read_manifest(entry["run_dir"]) or {}
    name = os.path.basename(entry["run_dir"].rstrip("/"))
    return {
        "modality": modality,
        "arch": "bert",
        "run": name,
        "run_dir": entry["run_dir"],
        "size": model_size(name, manifest)[0],
        "learning_rate": dig(manifest, "schedule.learning_rate"),
        "job_id": dig(manifest, "run.job_id"),
        "steps_run": None,
        "checkpoints_present": sorted(entry.get("checkpoints") or []),
    }


def collect(roots, read_args_seed=read_args_seed_from_checkpoint):
    """Every BERT run under ``roots``, as §3 rows."""
    rows = []
    for modality, root in roots:
        found, unreadable = walk_root(root, modality)
        tree = tree_label(root)
        for row in found:
            if row.get("arch") != "bert":
                continue
            rows.append(inventory_row(row, tree, read_args_seed=read_args_seed))
        for entry in unreadable:
            rows.append(inventory_row(
                row_without_trainer_state(entry, modality), tree,
                read_args_seed=read_args_seed,
                note="no readable trainer_state.json"))
    rows.sort(key=lambda r: (r["modality"], str(r["size"]), r["run"]))
    return rows


def provenance(roots, generated_by):
    """The convention's required header: the version line and the provenance block."""
    now = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9))
    ).replace(microsecond=0)
    lines = [
        f"#! report-tsv {RT_VERSION}",
        "# model: several (columns: modality,size,lr)",
        "# source: run_manifest.json per run; training_args.bin where there is no manifest",
        f"# generated: {now.isoformat()} by {generated_by}",
        "# note: config_seed is sources.seed.value (what the config asked for);"
        " manifest_seed is seed.seed (what transformers seeded with)",
        "# note: args_seed is filled only where there is no run_manifest.json,"
        " and comes from the newest checkpoint's training_args.bin",
    ]
    for modality, root in roots:
        lines.append(f"# note: root {modality}={tree_label(root)}")
    return lines


def write_tsv(path, rows, header_lines):
    """The file the convention describes: comment block, header, one row per run.

    No BOM and LF endings (convention §7, R5/R6): a BOM lands inside the first
    column's name. The reports under ``report/`` carry one; a TSV does not.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for line in header_lines:
            fh.write(line + "\n")
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")


def parse_roots(values):
    roots = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--root wants MODALITY=DIR, got {value!r}")
        modality, _, directory = value.partition("=")
        if not os.path.isdir(directory):
            raise SystemExit(f"not a directory: {directory}")
        roots.append((modality, directory))
    return roots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="append", required=True, metavar="MODALITY=DIR")
    ap.add_argument("--out", required=True, help="the TSV §3 asks for")
    args = ap.parse_args(argv)

    roots = parse_roots(args.root)
    rows = collect(roots)
    write_tsv(args.out, rows, provenance(roots, "scripts/bert_seed_inventory.py"))

    disagree = [r for r in rows if r["seed_agrees"] == "no"]
    no_manifest = [r for r in rows if r["manifest_seed"] == ""]
    recovered = [r for r in no_manifest if r["args_seed"] != ""]
    print(f"bert runs: {len(rows)}")
    print(f"  with a manifest seed       : {len(rows) - len(no_manifest)}")
    print(f"  without one                : {len(no_manifest)}"
          f" (args_seed recovered for {len(recovered)})")
    print(f"  config seed disagrees      : {len(disagree)}")
    for row in disagree:
        print(f"    {row['modality']} {row['size']} {row['run']}:"
              f" config {row['config_seed']} -> used {row['manifest_seed']}")
    print(f"  production                 : {sum(1 for r in rows if r['production'] == 'yes')}")
    without_history = [r for r in rows if r["note"]]
    if without_history:
        print(f"  no readable trainer_state  : {len(without_history)}")
    unknown = [r for r in rows
               if r["manifest_seed"] == "" and r["args_seed"] == ""]
    if unknown:
        print(f"  seed not established       : {len(unknown)}")
        for row in unknown:
            print(f"    {row['modality']} {row['run']}: {row['seed_source']}")
    print(f"=== outputs: {args.out} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
