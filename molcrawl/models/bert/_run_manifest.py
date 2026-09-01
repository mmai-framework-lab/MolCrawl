"""One human-readable file per run saying what the run actually was.

Every value here already exists somewhere. The problem is where: the batch
composition and the selection metric live in ``training_args.bin``, a pickle that
needs torch plus an unpickling shim to open and cannot be grepped or diffed, and
the rest -- what the data was, how it was masked, which eval rows were used, what
the degenerate baseline is -- lives only in the config file that produced the run.

That gap has cost real time. genome GPT-2 trained at an effective global batch of
640 while every document said 2,560, because ``batch_size`` is per-device and the
number nobody could see was the product. ``max_steps`` was derived once from a
``dataset_info.json`` that disagreed with the arrow files by 100,000 rows. The
adopted checkpoints were chosen on ``eval_loss_mask`` while HF marked the best
``eval_loss``, and the mismatch surfaced only at aggregation.

So this writes the derived values, not just the raw ones:

``batch.effective_global_batch``
    per_device x grad_accum x world_size, computed. The number that was wrong.
``schedule.decays_over_max_steps``
    HF has no separate decay length -- the schedule always runs to ``max_steps``,
    so truncating it compresses the decay. Recorded with the flag so a run that
    was cut short is visible rather than inferred.
``selection.judge_on`` beside ``selection.metric_for_best_model``
    Their disagreement is the thing to notice.
``collapse_detection.enabled``
    genome, RNA and protein all decided to leave this off. Without it in the run,
    a reader later cannot tell whether the detector was armed.
``data.rows_from``
    arrow vs dataset_info.json, since those disagree on this tree.
``placement.*``
    How many nodes and GPUs the scheduler actually handed out, beside the
    ``world_size`` the processes actually saw. GPUs are requested with ``--gpus=N``
    and the scheduler decides the node placement, so a request for 4 can arrive as
    2 nodes with 2 each. A launcher that hard-codes ``--nproc_per_node`` then runs
    on one node only, and ``world_size`` -- the multiplier in the effective batch --
    silently halves. ``placement.world_size_matches_allocation`` is that check,
    resolved at run time rather than left to be reconstructed from ``sacct`` later.

HF prints its own ``***** Running training *****`` banner with most of these
numbers, but at ``logger.info``, and transformers defaults to WARNING. Nothing in
this tree raises it, so across 855 job logs the banner appears zero times. The
values are echoed to stdout here instead of turning INFO on globally, which would
change the log volume of every running job to get six lines.
"""

import json
import os
from datetime import datetime, timezone

from molcrawl.models._provenance import (
    dirty_tree_warning,
    environment,
    git_state,
    value_sources,
)

MANIFEST = "run_manifest.json"

__all__ = ["MANIFEST", "TRACKED", "TRACKED_ENV", "dirty_tree_warning", "note_resume", "write_manifest"]

# Values whose provenance has mattered, reported with the default they would have
# had. HF splits these across TrainingArguments and the module globals, so the
# names are taken from whichever the configurator actually resolves.
TRACKED = (
    "learning_rate",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_length",
    "max_steps",
    "warmup_steps",
    "weight_decay",
    "save_steps",
    "eval_steps",
    "seed",
    "judge_on",
    "checkpoint_metric",
    "degenerate_loss_threshold",
    "document_masking",
)

# Read by the BERT configs. LEARNING_SOURCE_DIR, GENOME_SUBSET and
# HARD_MAX_STEPS_OVERRIDE are in _provenance.COMMON_ENV.
#
# SUBSET_BERT_EPOCHS is the one that has already gone unrecorded: genome's
# bert_small_subset.py derives max_steps from it, and the 9-epoch saturation run
# was launched by setting it -- a fact the run itself does not carry.
TRACKED_ENV = (
    "SUBSET_BERT_LR",
    "SUBSET_BERT_LARGE_LR",
    "SUBSET_BERT_EPOCHS",
    "SUBSET_BERT_MAX_CKPT",
    "BERT_LR_TAG",
    "SMOKE_MAX_STEPS",
    "SMOKE_WARMUP_STEPS",
    "SMOKE_EVAL_INTERVAL",
)


def _placement(world):
    """What the scheduler handed out, and whether the processes saw all of it."""
    def _int(name):
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
        # None when the allocation is unknown (not a SLURM job): unknown is not a
        # mismatch, and reporting False here would cry wolf on every local run.
        "world_size_matches_allocation": (
            None if gpus_total is None else gpus_total == world
        ),
    }


def write_manifest(output_dir, args, *, config, data, objective, evaluation,
                   resolved=None, defaults=None, resumed_from_step=None):
    """Write ``run_manifest.json`` into ``output_dir`` and return the dict.

    ``config`` is the curated dict the caller assembles (retention, collapse
    detection, epoch planning). ``resolved`` and ``defaults`` are main.py's whole
    globals snapshot after and before the configurator, which is what ``sources``
    is derived from -- the two are separate because only the second pair can say
    whether a value was chosen or merely inherited.
    """
    per_device = int(getattr(args, "per_device_train_batch_size", 0) or 0)
    accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    world = int(getattr(args, "world_size", 1) or 1)
    seq_len = evaluation.get("seq_len") or objective.get("seq_len")
    placement = _placement(world)

    manifest = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "git": git_state(),
            "output_dir": os.path.abspath(output_dir),
            "resumed_from_step": resumed_from_step,
        },
        "placement": placement,
        "data": data,
        "batch": {
            "per_device_train_batch_size": per_device,
            "gradient_accumulation_steps": accum,
            "world_size": world,
            "effective_global_batch": per_device * accum * world,
            "seq_len": seq_len,
            "tokens_per_step": (per_device * accum * world * seq_len) if seq_len else None,
        },
        "schedule": {
            "max_steps": int(getattr(args, "max_steps", 0) or 0),
            "warmup_steps": int(getattr(args, "warmup_steps", 0) or 0),
            "lr_scheduler_type": str(getattr(args, "lr_scheduler_type", "")),
            "learning_rate": float(getattr(args, "learning_rate", 0.0) or 0.0),
            # HF decays over max_steps; there is no separate decay length, so a
            # truncated max_steps compresses the schedule rather than ending it early.
            "lr_decay_steps": int(getattr(args, "max_steps", 0) or 0),
            "decays_over_max_steps": True,
            "max_steps_source": config.get("max_steps_source"),
            "epochs_planned": config.get("epochs_planned"),
        },
        "objective": objective,
        "eval": evaluation,
        "selection": {
            "metric_for_best_model": str(getattr(args, "metric_for_best_model", "")),
            "judge_on": evaluation.get("judge_on"),
            "save_steps": int(getattr(args, "save_steps", 0) or 0),
            "eval_steps": int(getattr(args, "eval_steps", 0) or 0),
            "save_total_limit": getattr(args, "save_total_limit", None),
            "retention": config.get("retention"),
        },
        "collapse_detection": config.get("collapse_detection"),
        "seed": {
            "seed": getattr(args, "seed", None),
            "data_seed": getattr(args, "data_seed", None),
        },
        "sources": value_sources(resolved or {}, defaults or {}, TRACKED),
        "env": environment(TRACKED_ENV),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return manifest


def note_resume(output_dir, step):
    """Record on an existing manifest that this run resumed, without losing the rest."""
    path = os.path.join(output_dir, MANIFEST)
    if not os.path.exists(path):
        return
    try:
        with open(path) as fh:
            m = json.load(fh)
        m.setdefault("run", {}).setdefault("resume_history", []).append(
            {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "from_step": step}
        )
        m["run"]["resumed_from_step"] = step
        with open(path, "w") as fh:
            json.dump(m, fh, indent=2, sort_keys=False)
            fh.write("\n")
    except (OSError, ValueError):
        pass
