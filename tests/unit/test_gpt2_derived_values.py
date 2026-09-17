"""gpt2 derived values: steps, epochs, learning-rate landmarks, checkpoint rule."""

import math

import pytest

from molcrawl.models.gpt2._run_manifest import derived_values


def nanogpt_lr(learning_rate, min_lr, warmup_iters, lr_decay_iters):
    """The same function as gpt2/train.py get_lr, closed over its settings."""
    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)
    return get_lr


def _derive(**over):
    kw = dict(
        train_rows=1_000_000, batch_size=16, gradient_accumulation_steps_configured=160, world_size=4,
        block_size=1024, max_iters=10_000, warmup_iters=1_000, lr_decay_iters=10_000, decay_lr=True,
        learning_rate=6e-4, min_lr=6e-5, eval_interval=500, save_checkpoint_steps=None,
        always_save_checkpoint=False, out_dir="rel/out",
    )
    kw.update(over)
    kw["lr_fn"] = nanogpt_lr(kw["learning_rate"], kw["min_lr"], kw["warmup_iters"], kw["lr_decay_iters"])
    return derived_values(**kw)


def test_steps_and_epochs():
    d = _derive()
    assert d["sequences_per_optimizer_step"] == 2560
    assert d["optimizer_steps_planned"] == 10_001
    assert d["steps_per_epoch"] == pytest.approx(1_000_000 / 2560)
    assert d["epochs_planned"] == pytest.approx(10_001 * 2560 / 1_000_000)
    assert d["tokens_per_optimizer_step"]["input_positions"] == 2560 * 1023
    assert d["tokens_per_optimizer_step"]["sequence_slots"] == 2560 * 1024


def test_learning_rate_landmarks_come_from_the_schedule():
    d = _derive()["learning_rate"]
    assert d["initial"] == 0.0
    assert d["peak"] == 6e-4 and d["peak_step"] == 1_000
    assert d["floor"] == 6e-5
    assert d["floor_reached_at_step"] == 10_000
    assert d["floor_reached_within_run"] is True
    assert d["at_final_step"] == pytest.approx(6e-5)


def test_floor_not_reached_when_decay_outlasts_the_run():
    d = _derive(lr_decay_iters=20_000)["learning_rate"]
    assert d["floor_reached_within_run"] is False
    assert d["at_final_step"] > 6e-5


def test_constant_schedule():
    d = _derive(decay_lr=False)
    assert d["learning_rate"]["schedule"] == "constant"
    assert d["learning_rate"]["at_final_step"] == 6e-4
    assert d["warmup_end_step"] == 0


def test_missing_rows_leave_epochs_null():
    d = _derive(train_rows=None)
    assert d["steps_per_epoch"] is None and d["epochs_planned"] is None


@pytest.mark.parametrize(
    "always,steps,expect",
    [
        (False, None, ["iter == max_iters", "a new best val loss"]),
        (True, None, ["every eval step", "iter == max_iters"]),
        (False, 2000, ["multiples of save_checkpoint_steps 2000", "a new best val loss"]),
    ],
)
def test_checkpoint_rule_lists_every_reason(always, steps, expect):
    rule = _derive(always_save_checkpoint=always, save_checkpoint_steps=steps)["intervals"]["checkpoint"]
    for part in expect:
        assert part in rule


def test_out_dir_is_absolute():
    import os
    assert os.path.isabs(_derive()["out_dir"])
