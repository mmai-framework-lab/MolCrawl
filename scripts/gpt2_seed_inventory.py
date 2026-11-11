#!/usr/bin/env python3
"""One row per GPT-2 run: the seed its config declared, and the seed train.py used.

Judgment `all-seed-judgment-2026-11-10` §5. The BERT inventory
(`scripts/bert_seed_inventory.py`) left GPT-2 out because nanoGPT seeds itself and
cannot be caught by the `TrainingArguments` fault the BERT inventory was built to
find. §5 asks for the GPT-2 side anyway, in the same shape: modality, size,
learning rate, run name, the seed the config declared, the value that reached
`torch.manual_seed`, and whether the run stands behind a result.

Where each column comes from, and why it is not one source:

* `manifest_seed` -- `seed.seed` in `run_manifest.json`: the value `train.py`
  resolved after the configurator ran. This is the value that was seeded with.
* `manual_seed_rank0` -- the same number. `train.py` calls
  `torch.manual_seed(seed + ddp_rank)`, so rank 0 gets `seed` itself and rank *k*
  gets `seed + k`. The offset rule is its own column rather than four rows.
* `config_seed` -- **not in the manifest.** A GPT-2 manifest records the resolved
  seed but not which config file produced it, and its `sources` block has no seed
  entry. The configurator prints `Overriding config with <path>:` followed by the
  file's text, so the declaration is in the job's log and nowhere else. That is
  what this script reads, by job id.
* `cli_seed` -- `Overriding: seed = N`, printed when the seed came from the command
  line instead of the config. A run with one of these was not steered by its config.

`train.py`'s module default was 1337 (nanoGPT's) until 2026-11-11, when the
tree-wide move to 42 changed it. A run whose log shows neither a config seed nor a
CLI seed fell back to whichever default was in force, so `config_seed` is left empty
and the note says so rather than guessing.

Run discovery is `audit_checkpoint_metric_match.walk_root`, and `is_production` is
the same function the checkpoint audit and the conditions deck use, so this file
cannot disagree with them about which directories are runs or which runs count.

Reads `run_manifest.json` and the job logs. No weight file is opened and nothing is
written inside a run directory.

Usage:
    python scripts/gpt2_seed_inventory.py \
        --root genome_sequence=<dir> --log-root <dir> --out <out.tsv>
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_checkpoint_metric_match import (  # noqa: E402
    dig,
    is_production,
    model_size,
    read_manifest,
    walk_root,
)

# The tree label comes from the BERT inventory rather than being written again, so
# the two TSVs name the same directory the same way. Four of the roots end in
# `compounds`, which is why a bare basename is not the label.
from bert_seed_inventory import tree_label  # noqa: E402

# The resolver the seed checker uses. A config that inherits its seed through a
# star-import does not have the number in its own text, so the log's echo of that
# one file cannot answer what it resolved to.
from check_seed_declarations import value_of  # noqa: E402

#: The report TSV convention (docs/07-reports/report_tsv_columns.md) version this
#: file claims in its name and in its first line.
RT_VERSION = 1

#: §5's columns first, in the order the judgment lists them, then what is needed to
#: say where each value came from. ASCII only: convention §6, decision 3.
COLUMNS = [
    "modality",
    "size",
    "lr",
    "run",
    "config_seed",
    "manual_seed_rank0",
    "per_rank_offset",
    "production",
    "manifest_seed",
    "cli_seed",
    "seed_agrees",
    "config",
    "seed_source",
    "job_id",
    "steps_run",
    "tree",
    "note",
]

#: What the configurator prints. `configurator.py` lines 36 and 56.
#
# Not anchored at the line start, and the path capture stops at `.py`. Four ranks
# print to the same file without a lock, so the lines arrive spliced: the log of
# job 87220 holds `Overriding: min_lr = 6e-05Overriding: seed = 1001Overriding:`
# on one line. An anchored match reads that line as nothing, which is how the five
# `retrain2-gpt2-*-s100x` runs first came out as config/run disagreements when the
# config was right and the splice was the problem.
LOG_CONFIG_PATH = re.compile(r"Overriding config with (\S+\.py):")
LOG_CLI_SEED = re.compile(r"Overriding: seed = (\d+)")
#: A seed declaration in the echoed config text. Anchored at the line start so an
#: indented assignment inside a conditional block, and `data_seed`, are not read as
#: the config's own declaration.
ECHOED_SEED = re.compile(r"^seed\s*=\s*(\d+)\s*$")
#: Slurm writes `%x-%j.out`, so the job id is the last run of digits in the name.
LOG_JOB_ID = re.compile(r"-(\d+)\.out$")


def index_logs(log_roots):
    """job id -> the log files that carry it.

    A job can have more than one file (a resumed run, a renamed job), so the value
    is a list and the caller decides. Sorted by modification time, newest last.
    """
    index = {}
    for root in log_roots:
        for dirpath, _dirs, filenames in os.walk(root):
            for name in filenames:
                match = LOG_JOB_ID.search(name)
                if not match:
                    continue
                index.setdefault(match.group(1), []).append(
                    os.path.join(dirpath, name))
    for paths in index.values():
        paths.sort(key=lambda p: os.path.getmtime(p))
    return index


def read_declared_seed(paths):
    """The config path, the config's seed and any CLI seed, out of a job's logs.

    Returns ``(config_path, config_seed, cli_seed, source)``. The configurator
    echoes the config once per rank, so the same declaration appears several times;
    they are collected and compared rather than taking the first, because two
    different values in one log would mean the run did not read what we think it
    read.
    """
    if not paths:
        return "", None, None, "no job log found"
    config_paths, config_seeds, cli_seeds = [], set(), set()
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    # A spliced line can carry more than one of these, so every
                    # pattern is tried against every line and none of them skips
                    # the rest.
                    config_paths.extend(m.group(1).strip()
                                        for m in LOG_CONFIG_PATH.finditer(line))
                    cli_seeds.update(int(m.group(1))
                                     for m in LOG_CLI_SEED.finditer(line))
                    match = ECHOED_SEED.match(line)
                    if match:
                        config_seeds.add(int(match.group(1)))
        except OSError as exc:
            return "", None, None, f"log unreadable: {exc.__class__.__name__}"
    config = ""
    for candidate in config_paths:
        # The path is absolute and names the checkout it ran from; the part under
        # molcrawl/ is what identifies the config.
        marker = "molcrawl/tasks/pretrain/configs/"
        config = candidate[candidate.index(marker) + len(marker):] \
            if marker in candidate else candidate
        break
    if len(config_seeds) > 1:
        return config, None, (min(cli_seeds) if cli_seeds else None), \
            f"log echoes more than one seed: {sorted(config_seeds)}"
    config_seed = next(iter(config_seeds)) if config_seeds else None
    cli_seed = max(cli_seeds) if cli_seeds else None
    if config_seed is None and cli_seed is None:
        return config, None, None, "job log declares no seed"
    return config, config_seed, cli_seed, "job log"


#: Where the config tree sits inside the repository.
CONFIG_ROOT = "molcrawl/tasks/pretrain/configs"


def git_show(commit, path):
    """A file's text at one commit, or None when that object is not here.

    Commits recorded before the history rewrite of 2026-10-20 can be gone, and a
    run launched from a worktree whose branch was deleted can be too. A missing
    object is an answer -- the config cannot be read back -- not an error.
    """
    try:
        done = subprocess.run(["git", "show", f"{commit}:{path}"],
                              capture_output=True, text=True, check=False)
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def resolve_seed_at_commit(commit, config_rel, seen=None):
    """``seed`` as the config tree at ``commit`` resolves it, imports followed.

    The same walk as ``check_seed_declarations.read``, reading from git instead of
    the working tree, because what matters is the tree the run was launched from
    and not today's. Returns ``(seed, how)``.
    """
    if not commit or not config_rel:
        return None, "no commit recorded"
    seen = set() if seen is None else seen
    path = f"{CONFIG_ROOT}/{config_rel}"
    if path in seen:
        return None, "import cycle"
    seen.add(path)
    text = git_show(commit, path)
    if text is None:
        return None, f"config not in the tree at {commit}"
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None, f"config at {commit} does not parse"
    inherited = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            parts = (node.module or "").split(".")
            if "configs" in parts:
                rel = "/".join(parts[parts.index("configs") + 1:]) + ".py"
                got, _how = resolve_seed_at_commit(commit, rel, seen)
                if got is not None:
                    inherited = got
    own, env = None, {}
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [x.id for x in node.targets if isinstance(x, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        got = value_of(node.value, env) if targets and node.value is not None else None
        for name in targets:
            if got is not None:
                env[name] = got
            if name == "seed":
                own = got
    if own is not None:
        return int(own), f"config tree at {commit}"
    if inherited is not None:
        return int(inherited), f"config tree at {commit} (inherited)"
    return None, f"config at {commit} declares no seed"


def inventory_row(row, tree, logs, note=""):
    """The §5 row for one GPT-2 run."""
    manifest = read_manifest(row["run_dir"]) or {}
    manifest_seed = dig(manifest, "seed.seed")
    offset = dig(manifest, "seed.per_rank_offset")
    config, config_seed, cli_seed, source = read_declared_seed(
        logs.get(str(row.get("job_id")), []))
    if config_seed is None and config:
        # The config inherits its seed, so the echo of that one file does not carry
        # it. Resolve it in the tree the run was launched from.
        commit = dig(manifest, "run.git.commit")
        config_seed, how = resolve_seed_at_commit(commit, config)
        source = how if config_seed is not None else f"{source}; {how}"
    declared = cli_seed if cli_seed is not None else config_seed
    if declared is None or manifest_seed is None:
        agrees = ""
    else:
        agrees = "yes" if int(declared) == int(manifest_seed) else "no"
    notes = [note]
    if manifest_seed is None:
        notes.append("no run_manifest.json: the seed this run used is not recorded")
    if cli_seed is not None:
        notes.append("seed came from the command line, not the config")
    return {
        "modality": row["modality"],
        "size": row.get("size") or model_size(row["run"], manifest)[0] or "",
        "lr": row.get("learning_rate") if row.get("learning_rate") is not None else "",
        "run": row["run"],
        "config_seed": "" if config_seed is None else config_seed,
        "manual_seed_rank0": "" if manifest_seed is None else manifest_seed,
        "per_rank_offset": offset or "",
        "production": "yes" if is_production(row) else "no",
        "manifest_seed": "" if manifest_seed is None else manifest_seed,
        "cli_seed": "" if cli_seed is None else cli_seed,
        "seed_agrees": agrees,
        "config": config,
        "seed_source": source,
        "job_id": row.get("job_id") or "",
        "steps_run": row.get("steps_run") or "",
        "tree": tree,
        "note": "; ".join(n for n in notes if n),
    }


def collect(roots, log_roots):
    """Every GPT-2 run under ``roots``, as §5 rows."""
    logs = index_logs(log_roots)
    rows = []
    for modality, root in roots:
        found, _unreadable = walk_root(root, modality)
        tree = tree_label(root)
        for row in found:
            if row.get("arch") != "gpt2":
                continue
            rows.append(inventory_row(row, tree, logs))
    rows.sort(key=lambda r: (r["modality"], str(r["size"]), r["run"]))
    return rows


def provenance(roots, log_roots, generated_by):
    """The convention's required header: the version line and the provenance block."""
    now = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9))
    ).replace(microsecond=0)
    lines = [
        f"#! report-tsv {RT_VERSION}",
        "# model: several (columns: modality,size,lr)",
        "# source: run_manifest.json per run for the seed used;"
        " the job log for what the config declared",
        f"# generated: {now.isoformat()} by {generated_by}",
        "# note: manual_seed_rank0 is the value train.py passed to"
        " torch.manual_seed on rank 0; rank k got that value plus k",
        "# note: config_seed comes from the configurator's echo of the config file"
        " in the job log, because a GPT-2 manifest does not name its config",
        "# note: cli_seed is filled when the log shows 'Overriding: seed = N',"
        " meaning the value came from the command line and not from the config",
        "# note: train.py's module default was 1337 until 2026-11-11 and is 42 now;"
        " a run with no declared seed fell back to the default of its day",
    ]
    for modality, root in roots:
        lines.append(f"# note: root {modality}={tree_label(root)}")
    for root in log_roots:
        lines.append(f"# note: log root {root}")
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
    ap.add_argument("--log-root", action="append", default=[], metavar="DIR",
                    help="a directory holding slurm logs named ...-<job id>.out")
    ap.add_argument("--out", required=True, help="the TSV §5 asks for")
    args = ap.parse_args(argv)

    roots = parse_roots(args.root)
    log_roots = [d for d in args.log_root if os.path.isdir(d)]
    missing = [d for d in args.log_root if not os.path.isdir(d)]
    rows = collect(roots, log_roots)
    write_tsv(args.out, rows,
              provenance(roots, log_roots, "scripts/gpt2_seed_inventory.py"))

    production = [r for r in rows if r["production"] == "yes"]
    no_manifest = [r for r in rows if r["manifest_seed"] == ""]
    no_config = [r for r in rows if r["config_seed"] == ""]
    disagree = [r for r in rows if r["seed_agrees"] == "no"]
    print(f"gpt2 runs: {len(rows)}")
    print(f"  production                 : {len(production)}")
    print(f"  with a manifest seed       : {len(rows) - len(no_manifest)}")
    print(f"  without one                : {len(no_manifest)}")
    print(f"  config seed not recoverable: {len(no_config)}"
          f" (of which production {sum(1 for r in no_config if r['production'] == 'yes')})")
    print(f"  config seed disagrees      : {len(disagree)}")
    for row in disagree:
        print(f"    {row['modality']} {row['size']} {row['run']}:"
              f" config {row['config_seed']} -> used {row['manifest_seed']}"
              f" (cli {row['cli_seed'] or '-'})")
    seeds = collections.Counter(r["manual_seed_rank0"] for r in production)
    print("  seeds across production    : "
          + ", ".join(f"{k or 'unknown'}:{v}" for k, v in sorted(
              seeds.items(), key=lambda kv: str(kv[0]))))
    if missing:
        print(f"  log roots that do not exist: {len(missing)}")
        for d in missing:
            print(f"    {d}")
    print(f"=== outputs: {args.out} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
