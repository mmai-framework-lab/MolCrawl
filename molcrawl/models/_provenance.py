"""The parts of a run manifest that are the same whichever trainer wrote it.

``models/gpt2/_run_manifest.py`` and ``models/bert/_run_manifest.py`` each write
what their own framework knows -- nanoGPT's GPU-independent global batch, HF's
selection metric and collapse detector. Three things are not framework-specific
at all, and every one of them exists because a value was resolved somewhere the
run never wrote down:

``git_state``
    A commit hash does not say the run came from that commit. An edited working
    tree runs happily and records the hash it was based on. Both trainers
    recorded only the hash.
``value_sources``
    Both trainers resolve config the same way: snapshot the module's globals,
    ``exec`` a config file over them, then let ``--key=value`` overwrite again --
    and the config files themselves read ``os.environ`` on the way. By the time
    anything is saved there is one number and no history. genome's production 21
    ran at learning_rate 1e-4 against a 6e-4 default, and it took opening 384
    checkpoints in 2026-08 to find that out.
``placement``
    GPUs are asked for with ``--gpus=N`` and the scheduler decides the node
    placement, so a request for 4 can arrive as 2 nodes with 2 each. Both
    launchers hard-code ``--nproc_per_node`` with ``--standalone`` and no
    ``MASTER_ADDR``, so they would drive one node only. What that costs differs
    by framework -- HF multiplies the effective batch by ``world_size`` and
    nanoGPT divides the accumulation by it -- but neither run said how many GPUs
    it was actually driving, so neither could be checked afterwards.
``environment``
    The overrides live in environment variables that neither ``ckpt.pt`` nor
    ``training_args.bin`` has ever carried. ``SUBSET_BERT_EPOCHS`` sets the epoch
    count genome's BERT schedule is derived from; the 9-epoch run used it and
    said so nowhere.

Each trainer passes its own names -- the values worth tracking and the variables
its configs read differ -- but the shape of the answer should not.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Iterable, Mapping, Optional

# How many changed paths to name before the list stops being useful in a manifest.
MAX_DIRTY_FILES = 50

# Read by config files across both trainers. Framework-specific names are added
# by the caller; these are the ones either side can hit.
COMMON_ENV = (
    "LEARNING_SOURCE_DIR",
    "GENOME_SUBSET",
    "HARD_MAX_STEPS_OVERRIDE",
)


def placement(world: int) -> Dict[str, Any]:
    """What the scheduler handed out, beside what the processes actually saw.

    ``world_size_matches_allocation`` is ``None``, not ``False``, when the
    allocation is unknown: outside SLURM there is nothing to disagree with, and
    ``False`` there would warn on every local run.
    """
    def _int(name: str) -> Optional[int]:
        try:
            return int(os.environ[name])
        except (KeyError, ValueError, TypeError):
            return None

    nodes = _int("SLURM_JOB_NUM_NODES") or _int("SLURM_NNODES")
    # --gpus=N sets SLURM_GPUS; SLURM_GPUS_ON_NODE counts only this node's share.
    gpus_total = _int("SLURM_GPUS")
    gpus_here = _int("SLURM_GPUS_ON_NODE")
    if gpus_total is None and gpus_here is not None and nodes is not None:
        gpus_total = gpus_here * nodes

    return {
        "nodes": nodes,
        "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "gpus_allocated": gpus_total,
        "gpus_on_this_node": gpus_here,
        "world_size": world,
        "world_size_matches_allocation": (
            None if gpus_total is None else gpus_total == world
        ),
    }


def _git(*args: str) -> str:
    try:
        done = subprocess.run(("git",) + args, capture_output=True, text=True, timeout=10)
        return done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state() -> Dict[str, Any]:
    """Commit and branch, plus whether the tree that produced this run was clean.

    ``dirty`` is the field that matters: with it False the commit identifies the
    code, and with it True the commit is only where the code started from.
    """
    dirty = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "--short", "HEAD") or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(dirty),
        "dirty_files": [line[3:] for line in dirty.splitlines()][:MAX_DIRTY_FILES],
    }


def dirty_tree_warning(state: Mapping[str, Any]) -> Optional[str]:
    """The line to print when a run starts from an edited checkout, or None.

    Deliberately a warning and not a refusal: the three workstreams launch from
    checkouts they have just edited, and failing here would block them. The
    manifest carries the file list either way, so a run that turns out to be odd
    can be checked against it afterwards.
    """
    if not state.get("dirty"):
        return None
    return (
        "Working tree had uncommitted changes at launch: this run is NOT "
        f"reproducible from commit {state.get('commit')}. "
        f"{len(state.get('dirty_files') or [])} file(s) differ; "
        "they are listed in run_manifest.json."
    )


def value_sources(
    config: Mapping[str, Any],
    defaults: Mapping[str, Any],
    tracked: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Per value: what it is, whether it is the default, and what the default was.

    ``defaults`` is the trainer's globals snapshotted before its configurator
    runs. That separates "something set this" from "nothing touched it" -- the
    distinction both checkpoint formats lose. It does not separate a config file
    from an environment variable from ``--key=value``: all three overwrite the
    same globals in place, and by the time we can look there is nothing left to
    tell them apart. ``resolved_by`` says so rather than guessing.

    A value equal to its default is reported as "default" rather than omitted, so
    the manifest states the agreement instead of leaving a reader to assume it.
    """
    sources: Dict[str, Dict[str, Any]] = {}
    for key in tracked:
        if key not in config:
            continue
        value = config[key]
        default = defaults.get(key)
        same = value == default
        sources[key] = {
            "value": value,
            "from": "default" if same else "overridden",
            "default": default,
            "resolved_by": None if same else "config file, env var or --key=value (indistinguishable)",
        }
    return sources


def environment(extra: Iterable[str] = ()) -> Dict[str, str]:
    """The tracked environment variables that are actually set.

    Absent names are left out rather than recorded as null, so the manifest shows
    what shaped the run and not the whole list of things that could have.
    """
    names = list(COMMON_ENV) + list(extra)
    return {name: os.environ[name] for name in names if name in os.environ}
