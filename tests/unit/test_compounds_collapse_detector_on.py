"""The four seed-42 compounds configs arm the collapse detector, and it fires on the real series.

The detector's stop is inert unless a config sets degenerate_loss_threshold. None of
these configs did, so when large seed 42 collapsed at the end of warmup it only got the
grad_norm warning and spent the rest of the schedule at the degenerate level.
"""

import ast
from pathlib import Path

import pytest

from molcrawl.models.bert._mlm_diagnostics import CollapseDetectionCallback

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "molcrawl" / "tasks" / "pretrain" / "configs" / "compounds").glob("bert_*_seed42.py"))

# The degenerate floor for this corpus, documented in models/bert/_mlm_diagnostics:
# unigram 2.635, unigram plus the copyable 10 per cent 2.395, left-neighbour 2.066.
DEGENERATE = 2.395


def _assigned(path, name):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    return None


def test_there_are_four_of_them():
    assert len(CONFIGS) == 4, [p.name for p in CONFIGS]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_each_arms_the_detector_at_the_degenerate_floor(path):
    assert _assigned(path, "degenerate_loss_threshold") == DEGENERATE


class _Control:
    def __init__(self):
        self.should_save = False
        self.should_training_stop = False


class _State:
    def __init__(self, step):
        self.global_step = step


def _args():
    return type("A", (), {"max_grad_norm": 1.0})()


# What large seed 42 (job 187295) actually logged once it had collapsed.
COLLAPSED = [2.4725, 2.4730, 2.4735, 2.4728, 2.5982, 2.6060, 2.6322, 2.6414]


def test_it_stops_on_the_series_large_actually_logged():
    cb = CollapseDetectionCallback(degenerate_threshold=DEGENERATE, patience=5, grace_steps=5000)
    control = _Control()
    step = 5100  # past the grace period, which is what keeps a healthy run alive
    for value in COLLAPSED:
        cb.on_log(_args(), _State(step), control, logs={"eval_loss_mask": value})
        step += 100
    assert control.should_training_stop is True


def test_the_grace_period_still_protects_the_early_plateau():
    """large sat at 2.47 from step 1,700; a healthy run can plateau there too."""
    cb = CollapseDetectionCallback(degenerate_threshold=DEGENERATE, patience=5, grace_steps=5000)
    control = _Control()
    step = 1700
    for value in COLLAPSED:
        cb.on_log(_args(), _State(step), control, logs={"eval_loss_mask": value})
        step += 100
    assert control.should_training_stop is False


def test_a_healthy_value_does_not_trip_it():
    cb = CollapseDetectionCallback(degenerate_threshold=DEGENERATE, patience=5, grace_steps=5000)
    control = _Control()
    step = 5100
    for value in [0.0710, 0.0699, 0.0705, 0.0702, 0.0708, 0.0701, 0.0699, 0.0700]:
        cb.on_log(_args(), _State(step), control, logs={"eval_loss_mask": value})
        step += 100
    assert control.should_training_stop is False
