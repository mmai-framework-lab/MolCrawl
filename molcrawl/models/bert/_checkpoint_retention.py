"""Keep the checkpoints worth keeping, ranked by the metric we actually judge on.

``save_total_limit`` evicts oldest-first and spares only the best of
``metric_for_best_model``. Anything selected on a different metric survives
purely by landing in the last few slots. In the genome BERT sweep that was
11 of 21 runs: the adopted checkpoint was chosen on ``eval_loss_mask`` while HF
protected the best ``eval_loss``, and it only still existed because it happened
to fall inside the window. A run whose best comes mid-training would have lost
it outright.

This callback replaces that with an explicit rule:

    keep the ``keep_best`` best checkpoints by ``metric``, plus the
    ``keep_latest`` most recent ones for resume

Set ``save_total_limit=None`` when this is active so the two do not both prune.

The metric has to have been logged at a step for that checkpoint to be ranked.
Evaluation and saving therefore need to happen on the same interval -- an eval
without a save is a candidate that was measured and thrown away, and a save
without an eval is a checkpoint that cannot be ranked and is only ever held by
``keep_latest``. Callers set the interval per modality; the guidance is roughly
100 eval points per run.
"""

import os
import shutil

from transformers import TrainerCallback

CHECKPOINT_PREFIX = "checkpoint-"


def _step_of(path):
    return int(os.path.basename(path).split("-")[1])


class BestNCheckpointRetention(TrainerCallback):
    """Prune checkpoints to the best N by ``metric`` plus the newest few."""

    def __init__(self, metric="eval_loss_mask", keep_best=10, keep_latest=1,
                 greater_is_better=False, fallback_metric="eval_loss"):
        self.metric = metric
        self.fallback_metric = fallback_metric
        self.keep_best = max(int(keep_best), 1)
        self.keep_latest = max(int(keep_latest), 1)
        self.greater_is_better = bool(greater_is_better)
        self._warned_fallback = False

    def _scores(self, state):
        """Latest value of the ranking metric at each step it was logged."""
        for name in (self.metric, self.fallback_metric):
            scores = {e["step"]: e[name] for e in state.log_history if name in e}
            if scores:
                if name != self.metric and not self._warned_fallback:
                    print(f"⚠️  checkpoint retention: {self.metric!r} never logged, "
                          f"ranking on {name!r} instead")
                    self._warned_fallback = True
                return scores
        return {}

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        existing = [
            os.path.join(args.output_dir, d)
            for d in os.listdir(args.output_dir)
            if d.startswith(CHECKPOINT_PREFIX)
            and os.path.isdir(os.path.join(args.output_dir, d))
            and d[len(CHECKPOINT_PREFIX):].isdigit()
        ]
        if len(existing) <= self.keep_best + self.keep_latest:
            return control

        by_step = {_step_of(p): p for p in existing}
        scores = self._scores(state)

        keep = set(sorted(by_step, reverse=True)[: self.keep_latest])

        # load_best_model_at_end reloads state.best_model_checkpoint when training
        # ends, so deleting it crashes the run at the finish line. That checkpoint
        # is the best of `metric_for_best_model`, which is a different metric from
        # the one ranking here: across the 21 genome BERT runs it fell outside the
        # eval_loss_mask top ten in 16 of them. Always keep it.
        best = getattr(state, "best_model_checkpoint", None)
        if best:
            try:
                keep.add(_step_of(best))
            except (IndexError, ValueError):
                pass
        ranked = sorted(
            (s for s in by_step if s in scores),
            key=lambda s: (-scores[s], s) if self.greater_is_better else (scores[s], s),
        )
        keep.update(ranked[: self.keep_best])

        # A checkpoint with no logged score cannot be ranked. Keeping it would
        # let unrankable checkpoints accumulate without bound, so drop it unless
        # keep_latest already holds it.
        for step in sorted(set(by_step) - keep):
            shutil.rmtree(by_step[step], ignore_errors=True)

        return control
