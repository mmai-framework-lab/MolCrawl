"""BERT early stopping decides on a metric that is added after evaluate() returns.

``MlmBreakdownMixin.evaluate`` adds ``eval_loss_mask`` after ``Trainer.evaluate``
has already fired ``on_evaluate``. HF's own callback therefore never saw it and
never stopped a run (job 355167). These tests reproduce that shape with the real
Trainer on CPU, lr 0 so the metric is constant, and check where each callback ends
the run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from torch import nn  # noqa: E402
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments  # noqa: E402

from molcrawl.models.bert._early_stopping import LoggedMetricEarlyStopping  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 1)

    def forward(self, x, labels):
        return {"loss": ((self.w(x).squeeze(-1) - labels) ** 2).mean()}


class Data(torch.utils.data.Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, i):
        return {"x": torch.tensor([float(i % 5), 1.0]), "labels": torch.tensor(float(i % 3))}


class AddsMetricAfterEvaluate(Trainer):
    """The shape of MlmBreakdownMixin: extra metric added and logged after super()."""

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        extra = {"eval_loss_mask": 2.5}
        metrics.update(extra)
        self.log(extra)
        return metrics


def run(tmp_path, callback, *, metric="eval_loss_mask", save_steps=10, trainer_cls=AddsMetricAfterEvaluate):
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=100,
        learning_rate=0.0,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=False,
        save_safetensors=False,
        report_to=[],
        use_cpu=True,
        seed=0,
    )
    trainer = trainer_cls(model=Tiny(), args=args, train_dataset=Data(), eval_dataset=Data(),
                          callbacks=[callback])
    trainer.train()
    return trainer.state.global_step


def test_hf_callback_never_stops_when_the_metric_arrives_after_on_evaluate(tmp_path):
    # The defect, reproduced: this is what every BERT run with early stopping on got.
    assert run(tmp_path, EarlyStoppingCallback(early_stopping_patience=3)) == 100


def test_logged_metric_callback_stops_patience_evals_after_the_best_is_saved(tmp_path):
    # 10: no best_metric yet (counter 0), the save at 10 sets it; 20/30/40 -> 3
    assert run(tmp_path, LoggedMetricEarlyStopping(early_stopping_patience=3)) == 40


def test_best_metric_is_only_written_at_a_save(tmp_path):
    # save every 20: 10 and 20 both see no best_metric; 30/40/50 -> 3
    assert run(tmp_path, LoggedMetricEarlyStopping(early_stopping_patience=3), save_steps=20) == 50


def test_fallback_to_eval_loss_is_decided_from_the_trainers_own_log(tmp_path):
    # main.py downgrades metric_for_best_model to eval_loss when the breakdown is
    # unavailable; Trainer.evaluate logs eval_loss itself.
    steps = run(tmp_path, LoggedMetricEarlyStopping(early_stopping_patience=3),
                metric="eval_loss", trainer_cls=Trainer)
    assert steps == 40


def _state(step, best=None):
    return SimpleNamespace(global_step=step, best_metric=best)


def _args():
    return SimpleNamespace(metric_for_best_model="eval_loss_mask", greater_is_better=False)


def _control():
    return SimpleNamespace(should_training_stop=False)


def test_training_loss_logs_are_ignored():
    cb = LoggedMetricEarlyStopping(early_stopping_patience=1)
    c = cb.on_log(_args(), _state(10, best=1.0), _control(), logs={"loss": 9.0, "learning_rate": 0.0})
    assert cb.early_stopping_patience_counter == 0 and not c.should_training_stop


def test_one_decision_per_evaluation_step():
    cb = LoggedMetricEarlyStopping(early_stopping_patience=2)
    args, st = _args(), _state(20, best=1.0)
    cb.on_log(args, st, _control(), logs={"eval_loss_mask": 1.0})
    cb.on_log(args, st, _control(), logs={"eval_loss_mask": 1.0})
    assert cb.early_stopping_patience_counter == 1


def test_on_evaluate_does_not_count():
    cb = LoggedMetricEarlyStopping(early_stopping_patience=1)
    c = cb.on_evaluate(_args(), _state(10, best=1.0), _control(), metrics={"eval_loss_mask": 1.0})
    assert cb.early_stopping_patience_counter == 0 and not c.should_training_stop
