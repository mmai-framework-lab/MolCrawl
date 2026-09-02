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

import json
import os
from datetime import datetime, timezone

from molcrawl.models._provenance import (
    dirty_tree_warning,
    environment,
    git_state,
    introduced_values,
    placement as _placement,
    value_sources,
)

MANIFEST = "run_manifest.json"

__all__ = ["MANIFEST", "TRACKED", "TRACKED_ENV", "dirty_tree_warning", "note_resume", "write_manifest"]

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


def write_manifest(out_dir, config, defaults, *, data, batch, schedule, objective,
                   evaluation, selection, seed, introduced=None,
                   configurator_path=None, resumed_from_iter=None):
    """Write ``run_manifest.json`` into ``out_dir`` and return the dict."""
    micro = int(batch.get("batch_size") or 0)
    accum = int(batch.get("gradient_accumulation_steps_configured") or 0)
    block = int(batch.get("block_size") or 0)
    effective = micro * accum

    manifest = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": "nanoGPT",
        "run": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "git": git_state(),
            "out_dir": os.path.abspath(out_dir),
            "resumed_from_iter": resumed_from_iter,
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
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return manifest


def note_resume(out_dir, iter_num):
    """Record on an existing manifest that this run resumed, without losing the rest."""
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(path):
        return
    try:
        with open(path) as fh:
            manifest = json.load(fh)
        manifest.setdefault("run", {}).setdefault("resume_history", []).append(
            {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "from_iter": iter_num}
        )
        manifest["run"]["resumed_from_iter"] = iter_num
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=False)
            fh.write("\n")
    except (OSError, ValueError):
        pass
