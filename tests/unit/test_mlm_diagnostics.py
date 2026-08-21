"""Tests for the MLM eval breakdown and the collapse detector.

The 2026-08-06 investigation showed that a single blended eval_loss hides a total
training failure: the 10% of positions whose answer stays visible in the input are
trivially copied, which pulls the average down. These tests pin the split so that
regression cannot creep back.
"""

from types import SimpleNamespace

import torch

from molcrawl.models.bert._mlm_diagnostics import (
    IGNORE_INDEX,
    CollapseDetectionCallback,
    MlmBreakdownMixin,
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


def _control():
    return type("C", (), {"should_training_stop": False})()


def _state(step):
    return type("S", (), {"global_step": step})()


def _args():
    return type("A", (), {"max_grad_norm": 1.0})()


def test_collapse_callback_fires_through_the_real_hook_order():
    """Trainer.evaluate() fires on_evaluate BEFORE the breakdown is added.

    MlmBreakdownMixin.evaluate() appends eval_loss_mask to the dict that
    super().evaluate() returns, but Trainer.evaluate() has already handed its own
    metrics to on_evaluate by then. The value only reaches a callback through the
    mixin's log(extra) call, so the detector has to work from on_log — it silently
    never fired for any modality until it did (protein sat 25 evals above its 2.63
    threshold without one warning).
    """
    cb = CollapseDetectionCallback(degenerate_threshold=2.63, patience=2, grace_steps=0)
    control = _control()

    for i, step in enumerate((100, 200), start=1):
        # what Trainer.evaluate() passes: no breakdown key yet
        cb.on_evaluate(_args(), _state(step), control, metrics={"eval_loss": 1.9})
        # what MlmBreakdownMixin.evaluate() logs immediately afterwards
        cb.on_log(_args(), _state(step), control, logs={"eval_loss_mask": 2.88})
        assert control.should_training_stop is (i == 2), f"eval {i} at step {step}"


def test_collapse_callback_counts_an_eval_once_across_both_hooks():
    """A single eval must not count twice when both hooks carry the metric."""
    cb = CollapseDetectionCallback(degenerate_threshold=2.63, patience=2, grace_steps=0)
    control = _control()

    cb.on_evaluate(_args(), _state(100), control, metrics={"eval_loss_mask": 2.88})
    cb.on_log(_args(), _state(100), control, logs={"eval_loss_mask": 2.88})
    assert control.should_training_stop is False  # one eval, not two

    cb.on_log(_args(), _state(200), control, logs={"eval_loss_mask": 2.88})
    assert control.should_training_stop is True


def test_grad_norm_warning_does_not_shadow_the_eval_check():
    """The old on_log returned early forever once grad_norm had been seen."""
    cb = CollapseDetectionCallback(degenerate_threshold=2.63, patience=1, grace_steps=0)
    control = _control()

    cb.on_log(_args(), _state(1), control, logs={"grad_norm": 4.8})
    cb.on_log(_args(), _state(100), control, logs={"eval_loss_mask": 2.88})
    assert control.should_training_stop is True


def test_grace_period_protects_the_plateau_every_healthy_run_crosses():
    """MLM sits at the degenerate level before breaking out.

    Measured plateaus: 1,500-2,000 steps (compounds seq 128, mol_nl) and 2,750
    steps (compounds packed with document masking). Stopping inside that window
    kills runs that were about to learn — which is what the old 3-eval default did
    at protein's log_interval=100, i.e. after 300 steps.
    """
    cb = CollapseDetectionCallback(degenerate_threshold=2.395, patience=3, grace_steps=5000)
    control = _control()

    # A run plateauing at the degenerate level right through the observed window.
    for step in range(100, 3000, 100):
        cb.on_evaluate(_args(), _state(step), control, metrics={"eval_loss_mask": 2.60})
    assert control.should_training_stop is False


def test_stops_once_past_the_grace_period():
    cb = CollapseDetectionCallback(degenerate_threshold=2.395, patience=3, grace_steps=5000)
    control = _control()

    for step in (5000, 5100, 5200):
        cb.on_evaluate(_args(), _state(step), control, metrics={"eval_loss_mask": 2.60})
    assert control.should_training_stop is True


def test_grace_period_can_be_disabled():
    cb = CollapseDetectionCallback(degenerate_threshold=2.395, patience=2, grace_steps=0)
    control = _control()

    cb.on_evaluate(_args(), _state(100), control, metrics={"eval_loss_mask": 2.60})
    cb.on_evaluate(_args(), _state(200), control, metrics={"eval_loss_mask": 2.60})
    assert control.should_training_stop is True


def test_breakout_during_grace_leaves_the_counter_clean():
    """A run that plateaus, breaks out, then wobbles must not carry stale counts."""
    cb = CollapseDetectionCallback(degenerate_threshold=2.395, patience=3, grace_steps=5000)
    control = _control()

    for step in range(100, 5000, 100):  # plateau, ignored
        cb.on_evaluate(_args(), _state(step), control, metrics={"eval_loss_mask": 2.60})
    cb.on_evaluate(_args(), _state(5000), control, metrics={"eval_loss_mask": 1.20})  # learning
    cb.on_evaluate(_args(), _state(5100), control, metrics={"eval_loss_mask": 2.60})
    cb.on_evaluate(_args(), _state(5200), control, metrics={"eval_loss_mask": 2.60})
    assert control.should_training_stop is False  # only 2 consecutive, patience is 3


def test_stop_is_announced_once(capsys):
    """Training stops at a boundary, so later evals must not re-announce it."""
    cb = CollapseDetectionCallback(degenerate_threshold=2.395, patience=1, grace_steps=0)
    control = _control()

    for step in (100, 200, 300):
        cb.on_evaluate(_args(), _state(step), control, metrics={"eval_loss_mask": 2.60})

    assert control.should_training_stop is True
    assert capsys.readouterr().out.count("STOPPING") == 1


class _RecordingModel:
    """Captures what the breakdown forward was given."""

    def __init__(self, vocab=16, seq=8):
        self.seen = None
        self.vocab, self.seq = vocab, seq

    def __call__(self, **kwargs):
        import torch

        self.seen = dict(kwargs)
        rows = kwargs["input_ids"].shape[0]
        return SimpleNamespace(logits=torch.zeros(rows, self.seq, self.vocab))


class _BareTrainer:
    """Stands in for the HF Trainer half of the mixin."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return (None, None, None)


class _Breakdown(MlmBreakdownMixin, _BareTrainer):
    pass


def test_the_breakdown_forward_gets_everything_the_collator_supplied():
    """position_ids must reach the model, or a document-masked run is scored wrongly.

    DocumentMaskingCollator resets position_ids per document. Forwarding only
    input_ids and attention_mask left the model reading 0..N-1 across a packed block
    it was trained to read per molecule, which inflated the reported [MASK] loss about
    six-fold - 0.869 against 0.141 on the W3 checkpoint - while HF's own eval_loss,
    computed by a proper forward, stayed at 0.129.
    """
    import torch

    trainer = _Breakdown()
    trainer.mlm_mask_token_id = 4
    model = _RecordingModel()
    inputs = {
        "input_ids": torch.tensor([[1, 4, 3, 4, 2, 5, 6, 7]]),
        "attention_mask": torch.ones(1, 8, 8, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, 9, -100, 8, -100, -100, -100, -100]]),
    }
    trainer.prediction_step(model, inputs, prediction_loss_only=True)

    assert "position_ids" in model.seen, "position_ids were dropped again"
    assert torch.equal(model.seen["position_ids"], inputs["position_ids"])
    assert torch.equal(model.seen["attention_mask"], inputs["attention_mask"])
    assert "labels" not in model.seen, "labels must not go into the forward"


def test_an_arm_without_position_ids_still_works():
    """1-molecule-per-row runs supply none; the forward must not require them."""
    import torch

    trainer = _Breakdown()
    trainer.mlm_mask_token_id = 4
    model = _RecordingModel()
    trainer.prediction_step(
        model,
        {
            "input_ids": torch.tensor([[1, 4, 3, 4, 2, 5, 6, 7]]),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "labels": torch.tensor([[-100, 9, -100, 8, -100, -100, -100, -100]]),
        },
        prediction_loss_only=True,
    )
    assert "position_ids" not in model.seen
