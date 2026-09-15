"""When the extra checkpoint is requested, and when it is not.

The callback exists because a minimum between two save steps leaves the reported
number without weights behind it. The cases that matter are the ones where it
must stay quiet: a metric that got worse, a metric that is not in the eval output
at all, and a step where HF was going to save anyway.
"""

from molcrawl.models.bert._save_on_improve import SaveOnMetricImprovement


class _Control:
    def __init__(self, should_save=False):
        self.should_save = should_save


class _State:
    def __init__(self, step):
        self.global_step = step


def _feed(cb, series, start_step=100, interval=100):
    """Run the callback over a series of metric values; return the steps it saved."""
    saved = []
    for i, value in enumerate(series):
        control = _Control()
        step = start_step + i * interval
        cb.on_evaluate(None, _State(step), control, metrics={"eval_loss_mask": value})
        if control.should_save:
            saved.append(step)
    return saved


def test_saves_only_where_the_metric_reaches_a_new_best():
    cb = SaveOnMetricImprovement(metric="eval_loss_mask")
    # Down, up, down past the previous best, flat, up.
    saved = _feed(cb, [1.00, 1.20, 0.80, 0.80, 0.90])
    assert saved == [100, 300]
    assert cb.best == 0.80
    assert cb.best_step == 300


def test_a_minimum_between_save_steps_is_not_lost():
    """The case the callback is for: the bottom of the curve is not on the grid."""
    cb = SaveOnMetricImprovement(metric="eval_loss_mask")
    series = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.6, 0.7, 0.8, 0.9]
    saved = _feed(cb, series)
    # step 600 carries the minimum and is not a multiple of the 1,000 save interval
    assert 600 in saved
    assert cb.best_step == 600
    assert [s for s in saved if s % 1000 == 0] == []


def test_a_step_hf_was_already_saving_is_left_alone():
    cb = SaveOnMetricImprovement(metric="eval_loss_mask")
    control = _Control(should_save=True)
    cb.on_evaluate(None, _State(1000), control, metrics={"eval_loss_mask": 0.5})
    assert control.should_save is True
    assert cb.best == 0.5  # still tracked, so the next improvement is measured from here


def test_falls_back_to_eval_loss_when_the_breakdown_is_missing():
    cb = SaveOnMetricImprovement(metric="eval_loss_mask", fallback_metric="eval_loss")
    control = _Control()
    cb.on_evaluate(None, _State(100), control, metrics={"eval_loss": 2.0})
    assert control.should_save is True
    assert cb.best == 2.0


def test_no_saves_when_neither_metric_is_reported():
    cb = SaveOnMetricImprovement(metric="eval_loss_mask", fallback_metric="eval_loss")
    control = _Control()
    cb.on_evaluate(None, _State(100), control, metrics={"eval_runtime": 1.0})
    assert control.should_save is False
    assert cb.best is None


def test_greater_is_better_reverses_the_comparison():
    cb = SaveOnMetricImprovement(metric="eval_accuracy", greater_is_better=True)
    saved = []
    for i, value in enumerate([0.5, 0.4, 0.7]):
        control = _Control()
        cb.on_evaluate(None, _State(100 + i * 100), control, metrics={"eval_accuracy": value})
        if control.should_save:
            saved.append(100 + i * 100)
    assert saved == [100, 300]
