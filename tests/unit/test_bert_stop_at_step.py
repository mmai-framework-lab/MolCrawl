"""StopAtStep ends the loop at its step and leaves that step evaluated and saved."""

from types import SimpleNamespace

import pytest

from molcrawl.models.bert._stop_at_step import StopAtStep


ARGS = SimpleNamespace(eval_strategy="steps", save_strategy="steps")


def _control():
    return SimpleNamespace(should_training_stop=False, should_evaluate=False, should_save=False)


def test_does_nothing_before_the_step():
    cb, control = StopAtStep(8000), _control()
    cb.on_step_end(ARGS, SimpleNamespace(global_step=7999, max_steps=40320), control)
    assert not (control.should_training_stop or control.should_evaluate or control.should_save)
    assert cb.stopped_at is None


def test_stops_at_the_step_with_an_evaluation_and_a_checkpoint():
    cb, control = StopAtStep(8000), _control()
    cb.on_step_end(ARGS, SimpleNamespace(global_step=8000, max_steps=40320), control)
    assert control.should_training_stop and control.should_evaluate and control.should_save
    assert cb.stopped_at == 8000


def test_leaves_evaluation_and_saving_off_when_the_run_has_them_off():
    cb, control = StopAtStep(5), _control()
    off = SimpleNamespace(eval_strategy="no", save_strategy="no")
    cb.on_step_end(off, SimpleNamespace(global_step=5, max_steps=40), control)
    assert control.should_training_stop
    assert not control.should_evaluate and not control.should_save


@pytest.mark.parametrize("step", [0, -1])
def test_refuses_a_step_that_would_never_arm(step):
    with pytest.raises(ValueError):
        StopAtStep(step)


def test_with_the_real_trainer_the_run_ends_early_on_the_full_schedule(tmp_path):
    # The point of the key: training ends at stop_at_step while the learning rate
    # at that step is the one the max_steps schedule gives -- not a schedule
    # squeezed into the shorter run.
    import torch
    from torch import nn
    from transformers import Trainer, TrainingArguments

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(2, 1)

        def forward(self, x, y):
            return {"loss": ((self.w(x).squeeze(-1) - y) ** 2).mean()}

    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            return {"x": torch.tensor([float(i), 1.0]), "y": torch.tensor(float(i % 3))}

    def run(stop):
        args = TrainingArguments(
            output_dir=str(tmp_path / f"stop{stop}"),
            max_steps=40,
            warmup_steps=8,
            learning_rate=1e-3,
            per_device_train_batch_size=4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            use_cpu=True,
            seed=0,
        )
        cbs = [StopAtStep(stop)] if stop else []
        trainer = Trainer(model=Tiny(), args=args, train_dataset=Data(), callbacks=cbs)
        trainer.train()
        lrs = {h["step"]: h["learning_rate"] for h in trainer.state.log_history if "learning_rate" in h}
        return trainer.state.global_step, lrs

    stopped_step, stopped_lr = run(12)
    full_step, full_lr = run(0)
    assert stopped_step == 12 and full_step == 40
    for step in range(1, 13):
        assert stopped_lr[step] == pytest.approx(full_lr[step])
