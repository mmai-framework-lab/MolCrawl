"""One human-readable file per run saying what the nanoGPT run actually was.

The BERT side got this in #154. The 244 nanoGPT runs on disk got nothing, and
the difference has cost real time three times over:

``learning_rate``
    genome's production 21 ran at 1e-4 while the config's default reads 6e-4 --
    ``gpt2_small_subset.py`` resolves it from ``SUBSET_GPT2_LR``, and the
    environment a run was launched with is written down nowhere. The value only
    surfaced by opening 384 ``ckpt.pt`` files in 2026-08.
``effective_global_batch``
    ``batch_size * gradient_accumulation_steps`` is the GPU-independent global
    batch here, because ``train.py`` divides the accumulation by the world size.
    Under HuggingFace semantics the same two names multiply by the world size
    instead. A config written in the HF reading shipped 8 x 80 as "2,560" when
    nanoGPT made it 640, and no artifact said which reading applied.
``dataset``
    ``ckpt.pt`` records ``dataset`` -- the modality name, "genome_sequence" --
    and never the directory. ``dataset_dir`` comes from ``LEARNING_SOURCE_DIR``
    and ``GENOME_SUBSET``, so which corpus a run read cannot be recovered from
    the run. Recomputing genome's epochs from the rows on disk today gives 2.71
    to 4.74 where ``max_iters`` was derived for exactly 3, and there is no way
    to tell whether the dataset changed or the run read a different one.

So this records the derived values, not just the raw ones, and it records where
each one came from:

``batch.effective_global_batch``
    ``batch_size * gradient_accumulation_steps``, captured before the world-size
    division, with ``gpu_independent`` stating that this is nanoGPT semantics.
``sources``
    for the values that have bitten us, whether the run took the module default
    or something overrode it, and what the default was. A value equal to the
    default is recorded as "default" rather than left silent, so agreement is
    visible and not merely assumed.
``data.dataset_dir`` / ``data.rows``
    the resolved directory and the row counts actually loaded, so the epoch
    arithmetic can be redone later against the corpus the run really used.
"""

import os
from datetime import datetime, timezone

from molcrawl.models._provenance import (
    SCHEMA_VERSION,
    RunLifecycle,
    config_file_record,
    configurator_locals,
    cpu_info,
    dirty_tree_warning,
    environment,
    environment_by_prefix,
    git_state,
    gpu_info,
    introduced_values,
    is_rank_zero,
    launcher_record,
    mark_used,
    names_read_by,
    now_iso,
    parse_argv,
    placement as _placement,
    runtime_versions,
    scalar_snapshot,
    slurm_info,
    stage_sources,
    update_json_atomic,
    value_sources,
    write_json_atomic,
)

MANIFEST = "run_manifest.json"

__all__ = [
    "MANIFEST",
    "TRACKED",
    "TRACKED_ENV",
    "build_provenance",
    "derived_values",
    "dirty_tree_warning",
    "note_resume",
    "start_run",
    "write_manifest",
]

# Values whose provenance has mattered. Each is reported with the default it
# would have had, so "the run took the default" is stated rather than implied.
TRACKED = (
    "learning_rate",
    "min_lr",
    "batch_size",
    "gradient_accumulation_steps",
    "block_size",
    "max_iters",
    "warmup_iters",
    "lr_decay_iters",
    "weight_decay",
    "grad_clip",
    "eval_interval",
    "eval_iters",
)

# Read by the nanoGPT configs. LEARNING_SOURCE_DIR and GENOME_SUBSET are in
# _provenance.COMMON_ENV, since the BERT configs read them too.
TRACKED_ENV = (
    "SUBSET_GPT2_LR",
    "SUBSET_GPT2_WD",
    "SUBSET_GPT2_MAX_CKPT",
    "GPT2_LR_TAG",
    "SMOKE_MAX_STEPS",
    "SMOKE_WARMUP_ITERS",
    "SMOKE_EVAL_INTERVAL",
)


def build_provenance(*, argv, trainer_file, configurator_path, defaults, after_config_file,
                     after_cli, resolved, after_deepspeed=None, read_sources=()):
    """Order §1.1-§1.3 and §3: the stages, the config file, and which keys code reads.

    ``defaults`` holds only the trainer's declared names; the later snapshots hold
    every scalar global, so a name a config introduced first appears at
    ``after_config_file``. configurator.py's own loop variables are removed from
    every snapshot, as ``introduced`` already does.
    """
    excluded = configurator_locals(configurator_path) if configurator_path else frozenset()
    clean = {
        name: (scalar_snapshot(snap, exclude=excluded) if snap is not None else None)
        for name, snap in (
            ("defaults", defaults),
            ("after_config_file", after_config_file),
            ("after_cli", after_cli),
            ("after_deepspeed", after_deepspeed),
            ("resolved", resolved),
        )
    }
    parsed = parse_argv(argv, trainer_file)
    final = clean["resolved"] or clean["after_cli"] or {}
    read = names_read_by(read_sources) if read_sources else None
    return {
        "argv": parsed["argv"],
        "cwd": os.getcwd(),
        "config_file": config_file_record(parsed),
        "cli_overrides": parsed["cli_overrides"],
        "argv_problems": parsed["problems"],
        "stages_recorded": [name for name, snap in clean.items() if snap is not None],
        "after_config_file_missing_reason": (
            None if after_config_file is not None
            else "configurator did not record _config_after_file (older configurator.py)"
        ),
        "keys": stage_sources(clean),
        "used": mark_used(final, read) if read is not None else None,
        "used_method": "static: names read by " + ", ".join(os.path.basename(s) for s in read_sources)
        if read_sources else None,
    }


def derived_values(*, train_rows, batch_size, gradient_accumulation_steps_configured, world_size,
                   block_size, max_iters, warmup_iters, lr_decay_iters, decay_lr, learning_rate,
                   min_lr, lr_fn, eval_interval, save_checkpoint_steps, always_save_checkpoint,
                   out_dir):
    """Order §1.4: values the namespace does not hold, each with what it was computed from.

    ``lr_fn`` is train.py's own ``get_lr``, called rather than re-derived here, so the
    recorded learning rates are the ones the loop uses.

    Two nanoGPT specifics are stated rather than smoothed over:

    - the loop runs iterations 0..max_iters inclusive, so it takes ``max_iters + 1``
      optimizer steps;
    - ``get_batch`` draws rows uniformly with replacement on every rank (numpy seeded
      ``seed + rank``), so an "epoch" here is the number of rows drawn divided by the
      rows in the split -- an expected-coverage equivalent, not a pass over the data.
    """
    micro = int(batch_size)
    accum = int(gradient_accumulation_steps_configured)
    sequences = micro * accum
    steps = int(max_iters) + 1
    rows = int(train_rows) if train_rows else None

    lr = {"schedule": "constant", "decay_lr": bool(decay_lr)}
    if decay_lr:
        floor_step = int(lr_decay_iters)
        lr.update({
            "schedule": "linear warmup from 0, cosine decay to min_lr, then min_lr",
            "initial": float(lr_fn(0)),
            "peak": float(learning_rate),
            "peak_step": int(warmup_iters),
            "floor": float(min_lr),
            "floor_reached_at_step": floor_step,
            "floor_reached_within_run": floor_step <= int(max_iters),
            "at_final_step": float(lr_fn(int(max_iters))),
            "formula": "it < warmup_iters: lr*it/warmup_iters; it > lr_decay_iters: min_lr; else cosine",
        })
    else:
        lr.update({"initial": float(learning_rate), "peak": float(learning_rate), "peak_step": 0,
                   "floor": float(learning_rate), "floor_reached_at_step": None,
                   "floor_reached_within_run": None, "at_final_step": float(learning_rate)})

    # train.py saves at an eval step (iter % eval_interval == 0 or iter == max_iters),
    # never at iter 0, when any of these holds -- the reasons add, they do not exclude.
    reasons = []
    if always_save_checkpoint:
        reasons.append("every eval step")
    if save_checkpoint_steps:
        reasons.append(f"eval steps that are multiples of save_checkpoint_steps {int(save_checkpoint_steps)}")
    reasons += ["iter == max_iters", "a new best val loss"]
    checkpoint_rule = "at an eval step, iter > 0, when any of: " + "; ".join(reasons)

    return {
        "sequences_per_optimizer_step": sequences,
        "sequences_per_optimizer_step_formula": f"batch_size {micro} x gradient_accumulation_steps (configured total) {accum}",
        "world_size": int(world_size),
        "tokens_per_optimizer_step": {
            "input_positions": sequences * (int(block_size) - 1),
            "sequence_slots": sequences * int(block_size),
            "note": "get_batch splits each block_size row into x = row[:-1], y = row[1:]",
        },
        "optimizer_steps_planned": steps,
        "optimizer_steps_planned_formula": f"max_iters {int(max_iters)} + 1 (iterations 0..max_iters)",
        "train_rows": rows,
        "steps_per_epoch": (rows / sequences) if rows and sequences else None,
        "epochs_planned": (steps * sequences / rows) if rows else None,
        "epoch_definition": "rows drawn / rows in the train split; rows are sampled with replacement",
        "warmup_end_step": int(warmup_iters) if decay_lr else 0,
        "learning_rate": lr,
        "intervals": {"eval": int(eval_interval), "checkpoint": checkpoint_rule},
        "out_dir": os.path.abspath(out_dir),
    }


def start_run(out_dir, *, run_id, argv, trainer_file, configurator_path, defaults,
              after_config_file, after_cli, init_from, purpose=None):
    """Write the ``starting`` manifest before any GPU work; return its lifecycle.

    Order §0: a run whose manifest cannot be written must fail before it trains,
    so the write error propagates. Non-zero ranks get a disabled lifecycle.
    """
    lifecycle = RunLifecycle(os.path.join(out_dir, MANIFEST), run_id, enabled=is_rank_zero())
    if not lifecycle.enabled:
        return lifecycle
    document = {
        "schema_version": SCHEMA_VERSION,
        "written": now_iso(),
        "framework": "nanoGPT",
        "architecture": "gpt2",
        "run": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "git": git_state(),
            "out_dir": os.path.abspath(out_dir),
            "purpose": purpose or os.environ.get("MOLCRAWL_RUN_PURPOSE") or "training",
        },
        "launcher": launcher_record(),
        "provenance": build_provenance(
            argv=argv, trainer_file=trainer_file, configurator_path=configurator_path,
            defaults=defaults, after_config_file=after_config_file, after_cli=after_cli,
            resolved=None,
        ),
        "slurm": slurm_info(),
        "env_by_prefix": environment_by_prefix(),
        "deepspeed": {"enabled": False},
    }
    lifecycle.start(document, resume_expected=(init_from == "resume"))
    return lifecycle


def write_manifest(out_dir, config, defaults, *, data, batch, schedule, objective,
                   evaluation, selection, seed, introduced=None,
                   configurator_path=None, resumed_from_iter=None,
                   lifecycle=None, provenance=None, batch_policy=None, model=None,
                   optimizer=None, launcher=None, deepspeed=None, purpose=None,
                   derived=None, collect_runtime=True):
    """Write ``run_manifest.json`` into ``out_dir`` and return the dict.

    Schema 2 adds sections beside the schema-1 fields; none of those is removed or
    renamed (order §2.10). With a ``lifecycle`` the document becomes that run's
    ``running`` record; without one it is written directly, as before.
    """
    micro = int(batch.get("batch_size") or 0)
    accum = int(batch.get("gradient_accumulation_steps_configured") or 0)
    block = int(batch.get("block_size") or 0)
    effective = micro * accum

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": "nanoGPT",
        "architecture": "gpt2",
        "run": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "git": git_state(),
            "out_dir": os.path.abspath(out_dir),
            "resumed_from_iter": resumed_from_iter,
            "purpose": purpose or os.environ.get("MOLCRAWL_RUN_PURPOSE") or "training",
        },
        "placement": _placement(int(batch.get("world_size") or 1)),
        "data": data,
        "batch": {
            "batch_size": micro,
            # As the config declared it. train.py divides this by the world size
            # for the inner loop; the product below is the same either way.
            "gradient_accumulation_steps_configured": accum,
            "gradient_accumulation_steps_per_rank": batch.get("gradient_accumulation_steps_per_rank"),
            "world_size": batch.get("world_size"),
            "effective_global_batch": effective,
            "gpu_independent": True,
            "seq_len": block,
            "tokens_per_step": effective * block if block else None,
        },
        "schedule": schedule,
        "objective": objective,
        "eval": evaluation,
        "selection": selection,
        "seed": seed,
        "sources": value_sources(config, defaults, TRACKED),
        # Scalars the config file added rather than overrode. Separate from
        # sources because there is no default to compare them against, and not
        # filtered against what is reported elsewhere -- an exclusion list is one
        # more list to forget to update, and a duplicated value costs nothing.
        "introduced": introduced_values(config, introduced or {}, configurator_path),
        "env": environment(TRACKED_ENV),
        # ---- schema 2 ---- #
        "env_by_prefix": environment_by_prefix(),
        "launcher": launcher if launcher is not None else launcher_record(),
        "provenance": provenance,
        "batch_policy": batch_policy,
        "derived": derived,
        "model": model,
        "optimizer": optimizer,
        "deepspeed": deepspeed if deepspeed is not None else {"enabled": False},
        "slurm": slurm_info(),
        "cpu": cpu_info(),
        "runtime": runtime_versions() if collect_runtime else None,
        "gpu": gpu_info() if collect_runtime else None,
    }

    if lifecycle is not None:
        return lifecycle.running(manifest)
    write_json_atomic(os.path.join(out_dir, MANIFEST), manifest)
    return manifest


def note_resume(out_dir, iter_num, lifecycle=None, **details):
    """Record on an existing manifest that this run resumed, without losing the rest.

    ``details`` (order §4.5) -- the checkpoint, world sizes before and after, the
    effective batch, whether optimizer state was restored -- are kept on the entry.
    """
    path = os.path.join(out_dir, MANIFEST)
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "from_iter": iter_num}
    entry.update(details)

    def _mutate(manifest):
        manifest.setdefault("run", {}).setdefault("resume_history", []).append(entry)
        manifest["run"]["resumed_from_iter"] = iter_num

    # With a lifecycle, update its in-memory document too: a later complete() or
    # fail() rewrites the file from that document and would drop this entry.
    try:
        if lifecycle is not None and lifecycle.document is not None:
            lifecycle.update(_mutate)
        else:
            update_json_atomic(path, _mutate)
    except (OSError, ValueError):
        pass
    return entry
