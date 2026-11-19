"""Score a checkpoint that already exists, without training it.

A number measured under one evaluation does not join a table measured under another.
The compounds batch-size grid fixes the masked positions (fixed_eval_mask), while the
existing 2,560 arms were measured with the positions redrawn at every evaluation, so
the two series do not connect -- and the only way to connect them is to measure the
existing checkpoints again under the grid's evaluation.

That is all this does: load one checkpoint into the trainer the ordinary launch has
already built, run one evaluation, and write the metrics with the conditions they were
measured under. Nothing is trained, nothing is saved, and the run being measured is not
written into -- the output goes to a directory the caller names.

The conditions travel with the number because they are what makes it comparable: the
collator reseeds from the batch's own bytes mixed with the seed, so the same positions
only come back when the rows are grouped the same way. The evaluation micro-batch, the
process count, the subset rows and the subset seed are therefore part of the
measurement rather than incidental to it, and a table that omits them cannot be
checked.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

#: Any one of these in a checkpoint directory means it holds weights.
WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def has_weights(checkpoint: str) -> bool:
    return any(os.path.exists(os.path.join(checkpoint, name)) for name in WEIGHT_FILES)


def guard(checkpoint: str, model_path: str, fixed_eval_mask: bool) -> None:
    """Refuse the measurement rather than produce a number that means something else."""
    if not checkpoint:
        raise SystemExit(
            "eval_only=True without eval_only_checkpoint. There is nothing to score: "
            "name the checkpoint directory to measure.")
    if not os.path.isdir(checkpoint):
        raise SystemExit(f"eval_only_checkpoint is not a directory: {checkpoint}")
    if not has_weights(checkpoint):
        raise SystemExit(
            f"eval_only_checkpoint holds no weights: {checkpoint}\n"
            f"  expected one of {', '.join(WEIGHT_FILES)}")
    if not fixed_eval_mask:
        # Without it the positions are redrawn here too, and the measurement would
        # reproduce the spread it exists to remove.
        raise SystemExit(
            "eval_only=True with fixed_eval_mask off. The re-measurement exists to put "
            "the existing checkpoints on the grid's fixed evaluation; redrawing the "
            "positions here would measure the same noise again. Set fixed_eval_mask "
            "and eval_mask_seed.")
    real_ckpt = os.path.realpath(checkpoint)
    real_out = os.path.realpath(model_path) if model_path else ""
    if real_out and (real_ckpt == real_out or real_ckpt.startswith(real_out + os.sep)):
        # The run being measured is the record the reported numbers came from.
        raise SystemExit(
            f"eval_only would write into the run it is measuring:\n"
            f"  checkpoint {real_ckpt}\n  output     {real_out}\n"
            "Point model_path at a separate directory.")


def load_weights(trainer, checkpoint: str) -> str:
    """Put the checkpoint's weights into the trainer's model.

    Trainer's own loader is the one used: it is what ``load_best_model_at_end`` uses,
    so a sharded checkpoint and the safetensors / bin difference are handled the same
    way the training path handles them. There is no fallback, because the failure that
    matters is the silent one -- scoring randomly initialised weights under a
    checkpoint's name would produce a plausible number that means nothing.
    """
    loader = getattr(trainer, "_load_from_checkpoint", None)
    if not callable(loader):
        raise SystemExit(
            "this Trainer has no _load_from_checkpoint, so the checkpoint's weights "
            "cannot be shown to have been loaded. Refusing to score instead of "
            "reporting a number from whatever weights the model happens to hold.")
    loader(checkpoint)
    return "Trainer._load_from_checkpoint"


def conditions(*, checkpoint: str, eval_batch: int, world_size: int,
               subset_rows: int, subset_random: bool, subset_seed: int,
               mask_seed: int, mask_rate: float, judged_on: str) -> Dict[str, Any]:
    """What the number is only comparable within."""
    return {
        "checkpoint": checkpoint,
        "step": checkpoint_step(checkpoint),
        "per_device_eval_batch_size": int(eval_batch),
        "processes": int(world_size),
        "eval_subset_rows": int(subset_rows),
        "eval_subset_random": bool(subset_random),
        "eval_subset_seed": int(subset_seed),
        "fixed_eval_mask": True,
        "eval_mask_seed": int(mask_seed),
        "mlm_probability": float(mask_rate),
        "judged_on": judged_on,
    }


def checkpoint_step(checkpoint: str):
    """The step in ``checkpoint-32000``, or None when the name does not carry one."""
    base = os.path.basename(os.path.normpath(checkpoint))
    stem, _, tail = base.rpartition("-")
    if stem == "checkpoint" and tail.isdigit():
        return int(tail)
    return None


def score(trainer, checkpoint: str, where: Dict[str, Any], out_path: str,
          log=print) -> Dict[str, Any]:
    """One evaluation of one checkpoint, written out with its conditions.

    Every rank runs the evaluation -- the metrics are reduced across them, and a rank
    that skipped it would leave the others waiting at the collective. Only rank 0
    writes.
    """
    how = load_weights(trainer, checkpoint)
    log(f"eval-only: weights from {checkpoint} via {how}")
    metrics = trainer.evaluate()
    record = dict(where)
    record["metrics"] = {k: v for k, v in sorted(metrics.items())}
    is_main = getattr(trainer, "is_world_process_zero", None)
    if callable(is_main) and not is_main():
        return record
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    log(f"eval-only: wrote {out_path}")
    for key in sorted(metrics):
        if key.startswith("eval_loss"):
            log(f"  {key} = {metrics[key]}")
    return record
