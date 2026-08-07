"""Tests for the MLM eval breakdown and the collapse detector.

The 2026-08-06 investigation showed that a single blended eval_loss hides a total
training failure: the 10% of positions whose answer stays visible in the input are
trivially copied, which pulls the average down. These tests pin the split so that
regression cannot creep back.
"""

import torch

from molcrawl.models.bert._mlm_diagnostics import (
    IGNORE_INDEX,
    CollapseDetectionCallback,
    split_mlm_loss,
)

MASK_ID = 14


def _one_hot_logits(targets, vocab=20, confidence=10.0):
    """Logits that predict `targets` almost surely, i.e. loss ~ 0."""
    logits = torch.zeros(*targets.shape, vocab)
    logits.scatter_(-1, targets.unsqueeze(-1).clamp(min=0), confidence)
    return logits


def test_splits_positions_by_input_type():
    # Three scored positions: [MASK], answer-visible (copy), random replacement.
    labels = torch.tensor([[IGNORE_INDEX, 5, 6, 7]])
    input_ids = torch.tensor([[3, MASK_ID, 6, 19]])  # 6 is visible, 19 is a wrong token
    logits = _one_hot_logits(labels.clamp(min=0))

    sums, counts = split_mlm_loss(logits, labels, input_ids, MASK_ID)

    assert counts == {"mask": 1, "copy": 1, "random": 1}
    for key in sums:
        assert sums[key] < 1e-3  # a perfect model scores ~0 everywhere


def test_ignores_unscored_positions():
    labels = torch.full((1, 4), IGNORE_INDEX)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    logits = torch.zeros(1, 4, 20)

    sums, counts = split_mlm_loss(logits, labels, input_ids, MASK_ID)

    assert counts == {"mask": 0, "copy": 0, "random": 0}
    assert sums == {"mask": 0.0, "copy": 0.0, "random": 0.0}


def test_copy_positions_are_easier_than_mask_positions():
    """Reproduce the degenerate model: it just echoes its input.

    That is the shape the real failure had — copy 0.029 against mask 2.453 on the
    compounds bert-small run.
    """
    labels = torch.tensor([[5, 6]])
    input_ids = torch.tensor([[MASK_ID, 6]])  # first is masked, second is visible
    logits = _one_hot_logits(input_ids)  # a model that only copies its input

    sums, counts = split_mlm_loss(logits, labels, input_ids, MASK_ID)

    assert counts["mask"] == 1 and counts["copy"] == 1
    assert sums["copy"] < 1e-3  # copying works
    assert sums["mask"] > 1.0  # the actual task does not


def test_collapse_callback_stops_after_patience():
    cb = CollapseDetectionCallback(degenerate_threshold=2.395, patience=2)
    control = type("C", (), {"should_training_stop": False})()

    cb.on_evaluate(None, None, control, metrics={"eval_loss_mask": 2.60})
    assert control.should_training_stop is False
    cb.on_evaluate(None, None, control, metrics={"eval_loss_mask": 2.55})
    assert control.should_training_stop is True


def test_collapse_callback_resets_when_recovering():
    cb = CollapseDetectionCallback(degenerate_threshold=2.395, patience=2)
    control = type("C", (), {"should_training_stop": False})()

    cb.on_evaluate(None, None, control, metrics={"eval_loss_mask": 2.60})
    cb.on_evaluate(None, None, control, metrics={"eval_loss_mask": 2.20})  # back under
    cb.on_evaluate(None, None, control, metrics={"eval_loss_mask": 2.60})
    assert control.should_training_stop is False  # not consecutive, so keep going


def test_collapse_callback_is_inert_without_threshold():
    cb = CollapseDetectionCallback(degenerate_threshold=None)
    control = type("C", (), {"should_training_stop": False})()

    for _ in range(10):
        cb.on_evaluate(None, None, control, metrics={"eval_loss_mask": 9.9})
    assert control.should_training_stop is False
