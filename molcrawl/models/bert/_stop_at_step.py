"""End training at a chosen step without changing the learning-rate schedule.

``max_steps`` does two jobs in HF Trainer: it ends the loop, and it is the length
the scheduler decays over. Setting it to 8,000 to look at the first 8,000 steps
of a 40,320-step run therefore trains a different run -- the same warmup, then a
decay eight times too steep. Stopping by hand keeps the schedule but leaves no
record of where the run was stopped.

This callback keeps ``max_steps`` for the scheduler and ends the loop itself.
4.45.1 checks the flag right after the step's log/evaluate/save:

    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    self._maybe_log_save_evaluate(...)
    ...
    if self.control.should_epoch_stop or self.control.should_training_stop:
        break

so raising ``should_training_stop`` in ``on_step_end`` ends training after that
step. ``should_evaluate`` and ``should_save`` are raised with it when the run
evaluates and saves at all, so the stop step carries an evaluation and a
checkpoint even when it is off the eval and save grid. A run with evaluation or
saving turned off is left that way: forcing an evaluation with no eval dataset
raises inside Trainer.

Off unless a config sets ``stop_at_step`` above 0. The scheduler, optimizer and
data order are not touched.
"""

from transformers import TrainerCallback


def _enabled(args, name):
    value = getattr(args, name, None)
    return value is not None and str(getattr(value, "value", value)) != "no"


class StopAtStep(TrainerCallback):
    """End the training loop once ``global_step`` reaches ``step``."""

    def __init__(self, step: int):
        if int(step) <= 0:
            raise ValueError(f"stop_at_step must be positive to be armed, got {step}")
        self.step = int(step)
        self.stopped_at = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step and self.stopped_at is None:
            self.stopped_at = state.global_step
            if _enabled(args, "eval_strategy"):
                control.should_evaluate = True
            if _enabled(args, "save_strategy"):
                control.should_save = True
            control.should_training_stop = True
            print(f"⏹️  stop_at_step: ending at step {state.global_step} (max_steps {state.max_steps} kept for the schedule)")
        return control
