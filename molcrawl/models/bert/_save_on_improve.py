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

``require_metric`` stops a run whose judged metric never arrives. It exists for one
case only: a config that names a metric of its own. ``eval_loss_mask`` is computed
for every BERT run and reaches ``on_log``, so the default name cannot go missing;
a name someone typed can, and then the callback would quietly never fire and the
run would keep only its grid checkpoints, with nothing in the log to show it. The
2026-11-10 judgement declined a guard for the default name on exactly that ground
-- a guard for a state that cannot happen -- and left this narrower one open.

The check waits for the second evaluation rather than the first. An evaluation logs
its own metrics, fires ``on_evaluate``, and only then does ``MlmBreakdownMixin`` add
the mask metric through ``self.log()``, so at the first ``on_evaluate`` a metric
that does arrive has not arrived yet. One full cycle has to complete before absence
means absence.
"""

from transformers import TrainerCallback


class SaveOnMetricImprovement(TrainerCallback):
    """Request a checkpoint whenever ``metric`` reaches a new best."""

    def __init__(self, metric="eval_loss_mask", greater_is_better=False,
                 fallback_metric=None, require_metric=False):
        self.metric = metric
        self.fallback_metric = fallback_metric
        self.greater_is_better = bool(greater_is_better)
        self.require_metric = bool(require_metric)
        self.best = None
        self.best_step = None
        self._seen_step = None
        self._warned_fallback = False
        self._warned_missing = False
        self._ever_seen = False
        self._eval_cycles = 0

    def _value(self, metrics):
        """The ranking metric for this report, or None if it is not in this one."""
        if metrics and self.metric in metrics:
            return float(metrics[self.metric])
        if self.fallback_metric and metrics and self.fallback_metric in metrics:
            # Only when a caller asked for one. Falling back by default is how a run
            # ends up selecting its best checkpoint on a metric it is not judged on.
            if not self._warned_fallback:
                print(f"⚠️  save-on-improve: {self.metric!r} absent, tracking "
                      f"{self.fallback_metric!r} as the caller asked")
                self._warned_fallback = True
            return float(metrics[self.fallback_metric])
        return None

    def _improves(self, value):
        if self.best is None:
            return True
        return value > self.best if self.greater_is_better else value < self.best

    def _consider(self, value, state, control):
        """Raise should_save if this value is a new best for this step."""
        if value is not None:
            self._ever_seen = True
        if value is None or self._seen_step == state.global_step:
            return control
        self._seen_step = state.global_step
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

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Where the breakdown metrics arrive.

        MlmBreakdownMixin.evaluate calls super().evaluate() first -- which is what
        fires on_evaluate -- and adds eval_loss_mask afterwards, through self.log().
        So a callback watching on_evaluate never sees the mask metric and, with a
        default fallback, silently tracked eval_loss instead; measured on compounds
        BERT medium (job 151323), where the eval_loss_mask minimum at step 14,700
        was therefore never saved. self.log() still runs inside _evaluate, before
        the Trainer tests control.should_save, so raising the flag here works.
        """
        return self._consider(self._value(logs), state, control)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Kept for metrics that do arrive here, such as a plain eval_loss."""
        control = self._consider(self._value(metrics), state, control)
        self._eval_cycles += 1
        self._require_arrival(state)
        return control

    def _require_arrival(self, state):
        """Stop a run whose named metric has not arrived after a full cycle.

        Only when the caller asked. The run is a config error, not a training
        failure: every checkpoint it would go on to write is chosen by the save
        grid alone, and the number the ladder reports would not be one of them.
        """
        if not self.require_metric or self._ever_seen or self._eval_cycles < 2:
            return
        raise RuntimeError(
            f"save-on-improve: {self.metric!r} has not appeared in any metrics or "
            f"log after {self._eval_cycles} evaluations (step {state.global_step}). "
            "The name comes from this run's config, so it is a configuration error: "
            "no extra checkpoint would ever be written and the best checkpoint would "
            "be whichever one the save grid happened to catch. Fix the metric name, "
            "or drop save_on_improve_metric to follow judge_on."
        )
