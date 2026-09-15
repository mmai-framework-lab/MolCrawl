"""Save a checkpoint at the step the judged metric improves, not only on the save grid.

HF Trainer saves on its own interval. ``best_model_checkpoint`` is then chosen
from the checkpoints that happen to exist, so the checkpoint we report and the
one downstream evaluation reads are the same only when the minimum lands on a
save step.

Run 53767 got away with it: evaluation ran every 500 steps, saving every 2,500,
and the minimum fell at step 15,000 -- the final step, which was also a save
point, on a curve that never turned back up. The GPT-2 ladder on the same corpus
does turn: at matched length the large arm bottomed at step 1,850 of 4,674. With
evaluation every 100 steps and saving every 1,000, a minimum that falls between
save points is the ordinary case, not the exception, and nine times in ten the
weights behind the reported number no longer exist when the run ends.

``transformers`` 5.x has ``save_strategy="best"`` for this, but it replaces the
periodic saves rather than adding to them, and 4.45.1 -- the version this project
runs on -- does not have it at all. What 4.45.1 does do is evaluate before it
tests ``control.should_save``:

    metrics = self._evaluate(...)          # fires on_evaluate
    if self.control.should_save:
        self._save_checkpoint(model, trial, metrics=metrics)

so a callback that raises the flag during ``on_evaluate`` gets a save at that
step, and the periodic ones still happen. ``_save_checkpoint`` receives the same
metrics and updates ``best_metric`` / ``best_model_checkpoint`` from them, so the
recorded best points at a checkpoint that exists.

Off unless a config sets ``save_on_improve = True``: it changes which
checkpoints a run writes, and no run in flight should have that move under it.
"""

from transformers import TrainerCallback


class SaveOnMetricImprovement(TrainerCallback):
    """Request a checkpoint whenever ``metric`` reaches a new best."""

    def __init__(self, metric="eval_loss_mask", greater_is_better=False,
                 fallback_metric="eval_loss"):
        self.metric = metric
        self.fallback_metric = fallback_metric
        self.greater_is_better = bool(greater_is_better)
        self.best = None
        self.best_step = None
        self._warned_fallback = False
        self._warned_missing = False

    def _value(self, metrics):
        """The ranking metric for this evaluation, or None if it was not logged."""
        for name in (self.metric, self.fallback_metric):
            if metrics and name in metrics:
                if name != self.metric and not self._warned_fallback:
                    print(f"⚠️  save-on-improve: {self.metric!r} not in the eval metrics, "
                          f"tracking {name!r} instead")
                    self._warned_fallback = True
                return float(metrics[name])
        if not self._warned_missing:
            print(f"⚠️  save-on-improve: neither {self.metric!r} nor "
                  f"{self.fallback_metric!r} is in the eval metrics; no extra saves")
            self._warned_missing = True
        return None

    def _improves(self, value):
        if self.best is None:
            return True
        return value > self.best if self.greater_is_better else value < self.best

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        value = self._value(metrics)
        if value is None:
            return control
        if not self._improves(value):
            return control

        self.best = value
        self.best_step = state.global_step
        # Already saving at this step for another reason -- leave the flag alone
        # rather than reporting a save we did not cause.
        if not control.should_save:
            control.should_save = True
            print(f"💾 {self.metric} improved to {value:.4f} at step "
                  f"{state.global_step}; saving off the interval")
        return control
