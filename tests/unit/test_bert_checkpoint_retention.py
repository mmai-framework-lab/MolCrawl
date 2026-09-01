"""What the retention callback keeps, and how many.

The question this answers came up when planning a run split across five jobs: the
ranking is rebuilt from ``state.log_history`` every time, and log_history holds an
entry for every evaluation -- including evaluations of checkpoints that have since
been deleted. If a deleted step could enter the top N, the number of checkpoints
actually on disk would drift below N and a run's best model could be a directory
that no longer exists.
"""

import json
import os

import pytest

from molcrawl.models.bert._checkpoint_retention import BestNCheckpointRetention


class _State:
    def __init__(self, log_history, best=None, step=0):
        self.log_history = log_history
        self.best_model_checkpoint = best
        self.global_step = step
        self.is_world_process_zero = True


class _Args:
    def __init__(self, output_dir):
        self.output_dir = output_dir


def _make(tmp_path, steps):
    for s in steps:
        (tmp_path / f"checkpoint-{s}").mkdir()
    return _Args(str(tmp_path))


def _on_disk(tmp_path):
    return sorted(
        int(d.name.split("-")[1]) for d in tmp_path.iterdir() if d.name.startswith("checkpoint-")
    )


def test_a_deleted_step_cannot_re_enter_the_kept_set(tmp_path):
    """log_history outlives the directories, so ranking must intersect with disk."""
    steps = [1000, 2000, 3000]
    args = _make(tmp_path, steps)
    # 500 scored better than anything still present, and is already gone.
    history = [{"step": 500, "eval_loss_mask": 0.01}] + [
        {"step": s, "eval_loss_mask": 0.5 - s / 1e5} for s in steps
    ]

    cb = BestNCheckpointRetention(keep_best=1, keep_latest=1)
    cb.on_save(args, _State(history, step=3000), None)

    kept = _on_disk(tmp_path)
    assert 500 not in kept
    # keep_best 1 (step 3000, the lowest loss present) + keep_latest 1 (also 3000)
    assert kept == [3000]


def test_a_run_that_keeps_improving_retains_keep_best_not_keep_best_plus_one(tmp_path):
    """The kept set is a union, so keep_latest costs nothing when it is already best.

    Planning disk from ``keep_best + keep_latest`` is therefore an upper bound, not
    the count. A monotonically improving run -- which is what a healthy pretrain
    looks like -- ends with the newest checkpoint inside the top N, and N on disk.
    """
    steps = list(range(1000, 16000, 1000))  # 15 checkpoints, each better than the last
    args = _make(tmp_path, steps)
    history = [{"step": 250, "eval_loss_mask": 0.001}]  # deleted, and the best ever
    history += [{"step": s, "eval_loss_mask": 1.0 - s / 1e5} for s in steps]

    cb = BestNCheckpointRetention(keep_best=10, keep_latest=1)
    cb.on_save(args, _State(history, step=15000), None)

    kept = _on_disk(tmp_path)
    assert kept == list(range(6000, 16000, 1000))
    assert len(kept) == 10
    assert 250 not in kept


def test_a_late_regression_costs_one_extra_checkpoint(tmp_path):
    """When the newest is not among the best, keep_latest adds to the union."""
    steps = list(range(1000, 16000, 1000))
    args = _make(tmp_path, steps)
    # Improving until 14000, then the last eval is the worst of the run.
    history = [{"step": s, "eval_loss_mask": 1.0 - s / 1e5} for s in steps[:-1]]
    history.append({"step": 15000, "eval_loss_mask": 9.0})

    cb = BestNCheckpointRetention(keep_best=10, keep_latest=1)
    cb.on_save(args, _State(history, step=15000), None)

    kept = _on_disk(tmp_path)
    assert len(kept) == 11
    assert kept[-1] == 15000


def test_the_metric_the_run_is_judged_on_decides_what_survives(tmp_path):
    """Ranking on eval_loss would keep a different set; that mismatch was PR #155."""
    steps = [1000, 2000, 3000]
    args = _make(tmp_path, steps)
    history = [
        {"step": 1000, "eval_loss_mask": 0.9, "eval_loss": 0.1},
        {"step": 2000, "eval_loss_mask": 0.2, "eval_loss": 0.9},
        {"step": 3000, "eval_loss_mask": 0.8, "eval_loss": 0.8},
    ]

    cb = BestNCheckpointRetention(metric="eval_loss_mask", keep_best=1, keep_latest=1)
    cb.on_save(args, _State(history, step=3000), None)

    # 2000 is best on eval_loss_mask; 1000 is best on eval_loss and must not survive.
    assert _on_disk(tmp_path) == [2000, 3000]


def test_the_best_model_checkpoint_survives_even_outside_the_top_n(tmp_path):
    """load_best_model_at_end reloads it at the end; deleting it crashes the finish."""
    steps = [1000, 2000, 3000]
    args = _make(tmp_path, steps)
    history = [{"step": s, "eval_loss_mask": 0.1 * i} for i, s in enumerate(steps, 1)]

    cb = BestNCheckpointRetention(keep_best=1, keep_latest=1)
    cb.on_save(
        args,
        _State(history, best=os.path.join(str(tmp_path), "checkpoint-2000"), step=3000),
        None,
    )

    assert 2000 in _on_disk(tmp_path)
