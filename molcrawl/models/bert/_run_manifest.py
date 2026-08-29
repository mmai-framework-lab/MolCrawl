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
"""

import json
import os
from datetime import datetime, timezone

MANIFEST = "run_manifest.json"


def _git_commit():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def write_manifest(output_dir, args, *, config, data, objective, evaluation,
                   resumed_from_step=None):
    """Write ``run_manifest.json`` into ``output_dir`` and return the dict."""
    per_device = int(getattr(args, "per_device_train_batch_size", 0) or 0)
    accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    world = int(getattr(args, "world_size", 1) or 1)
    seq_len = evaluation.get("seq_len") or objective.get("seq_len")

    manifest = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "commit": _git_commit(),
            "output_dir": os.path.abspath(output_dir),
            "resumed_from_step": resumed_from_step,
        },
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
