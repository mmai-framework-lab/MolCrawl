"""Early stopping that waits for the metric to be logged, not for on_evaluate.

HF's ``EarlyStoppingCallback`` reads ``metric_for_best_model`` in ``on_evaluate``,
which ``Trainer.evaluate`` fires *inside* the call. ``MlmBreakdownMixin.evaluate``
adds ``eval_loss_mask`` (and ``_copy`` / ``_random``) only after that call returns,
so the callback never saw the metric this project judges on: it logged
"did not find eval_loss_mask so early stopping is disabled" at every evaluation
and returned. Measured end to end in job 355167 (``workflows/early-stopping-probe
.sbatch``): four BERT cases with early stopping on all ran to max_steps, with the
patience counter at 0 in every trainer_state.json.

The mixin hands the breakdown to ``self.log``, so it reaches ``on_log``. This
subclass makes its decision there instead, once per evaluation step, and leaves
the rule itself to HF (``check_metric_value``: strictly better than
``state.best_metric`` by more than the threshold resets the counter, anything else
adds one, stop at ``patience``). The timing relative to the checkpoint save is
unchanged -- both hooks run before ``_save_checkpoint`` writes ``best_metric`` --
so a stop lands on the same step HF's own callback would choose had it seen the
metric.

``on_log`` also carries the plain ``eval_loss`` dict that ``Trainer.evaluate`` logs
itself, so a run whose ``metric_for_best_model`` falls back to ``eval_loss`` is
decided from that one. Training-loss dicts carry neither and are ignored.

``on_train_begin`` exists for one case: turning early stopping ON for a run whose
checkpoints were written with it OFF. ``Trainer._save_checkpoint`` writes every
``ExportableState`` callback into ``state.stateful_callbacks`` and indexes that dict
by class name with no default, while a resume restores the dict from the checkpoint
-- which, for a run that had no such callback, has no entry for this one. The first
save after the resume then dies with ``KeyError: 'LoggedMetricEarlyStopping'``
*after* the step was trained and evaluated. Measured on the molnl BERT extensions of
2026-11-16: jobs 355708 and 355711 both reached step 24,100, improved on the value
they resumed from, and crashed on writing that improvement.
"""

from __future__ import annotations

from transformers import EarlyStoppingCallback


class LoggedMetricEarlyStopping(EarlyStoppingCallback):
    """``EarlyStoppingCallback`` that decides when the metric is logged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decided_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Make sure this callback has a slot in the state the next save will write.

        ``setdefault``, not assignment: a resume that did restore this callback's own
        entry keeps the counter it had, and the live value is written over it at the
        next save either way.
        """
        callbacks = getattr(state, "stateful_callbacks", None)
        if isinstance(callbacks, dict):
            callbacks.setdefault(type(self).__name__, self.state())
        return control

    def _metric_key(self, args):
        key = args.metric_for_best_model
        return key if key.startswith("eval_") else f"eval_{key}"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # The metric may not be in ``metrics`` yet (see the module docstring); the
        # decision is made in on_log, which every evaluation reaches.
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        key = self._metric_key(args)
        if not logs or key not in logs:
            return control
        if self._decided_step == state.global_step:
            return control
        self._decided_step = state.global_step
        self.check_metric_value(args, state, control, float(logs[key]))
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            print(f"⏹️  Early stopping at step {state.global_step}: {key} has not improved on "
                  f"{state.best_metric} for {self.early_stopping_patience_counter} evaluations")
            control.should_training_stop = True
        return control
