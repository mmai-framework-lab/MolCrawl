"""Scoring a checkpoint that already exists, without training it.

The re-measurement of the existing 2,560 arms only means anything if it is refused
when it would mean something else: a checkpoint with no weights, a redrawn mask, or an
output directory inside the run being measured. Those three refusals are the cases
here, together with the record the measurement leaves -- the conditions have to travel
with the number, because the fixed mask only repeats while the rows are grouped the
same way.

Nothing here loads torch or transformers: the trainer is a stand-in with the two
methods the module calls.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    path = repo_root() / "molcrawl" / "models" / "bert" / "_eval_only.py"
    spec = importlib.util.spec_from_file_location("_eval_only", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTrainer:
    """Only what the module calls: load the weights, evaluate, say which rank."""

    def __init__(self, metrics=None, main=True):
        self.metrics = metrics or {"eval_loss": 0.0712, "eval_loss_mask": 0.2673}
        self._main = main
        self.loaded = None
        self.evaluated = 0

    def _load_from_checkpoint(self, path):
        self.loaded = path

    def evaluate(self):
        self.evaluated += 1
        return dict(self.metrics)

    def is_world_process_zero(self):
        return self._main


@pytest.fixture
def checkpoint(tmp_path):
    ckpt = tmp_path / "run" / "checkpoint-32000"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"")
    return ckpt


# --- what it refuses ------------------------------------------------------------

def test_a_checkpoint_must_be_named(mod, tmp_path):
    with pytest.raises(SystemExit, match="nothing to score"):
        mod.guard("", str(tmp_path / "out"), True)


def test_a_missing_directory_is_refused(mod, tmp_path):
    with pytest.raises(SystemExit, match="not a directory"):
        mod.guard(str(tmp_path / "nope"), str(tmp_path / "out"), True)


def test_a_directory_without_weights_is_refused(mod, tmp_path):
    empty = tmp_path / "checkpoint-1"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no weights"):
        mod.guard(str(empty), str(tmp_path / "out"), True)


@pytest.mark.parametrize("name", ["model.safetensors", "pytorch_model.bin",
                                  "model.safetensors.index.json",
                                  "pytorch_model.bin.index.json"])
def test_every_weight_layout_counts_as_weights(mod, tmp_path, name):
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / name).write_bytes(b"")
    assert mod.has_weights(str(ckpt))
    mod.guard(str(ckpt), str(tmp_path / "out"), True)


def test_a_redrawn_mask_is_refused(mod, checkpoint, tmp_path):
    # The whole point is to put the old checkpoints on the grid's fixed evaluation.
    with pytest.raises(SystemExit, match="fixed_eval_mask"):
        mod.guard(str(checkpoint), str(tmp_path / "out"), False)


def test_writing_into_the_run_being_measured_is_refused(mod, checkpoint):
    run = checkpoint.parent
    with pytest.raises(SystemExit, match="write into the run"):
        mod.guard(str(checkpoint), str(run), True)


def test_the_checkpoint_itself_as_output_is_refused(mod, checkpoint):
    with pytest.raises(SystemExit, match="write into the run"):
        mod.guard(str(checkpoint), str(checkpoint), True)


def test_a_separate_directory_is_accepted(mod, checkpoint, tmp_path):
    mod.guard(str(checkpoint), str(tmp_path / "reval"), True)


def test_a_sibling_name_sharing_a_prefix_is_not_inside(mod, checkpoint, tmp_path):
    # "run-reval" starts with "run" as a string but is a different directory.
    mod.guard(str(checkpoint), str(checkpoint.parent.parent / "run-reval"), True)


# --- the measurement ------------------------------------------------------------

def test_weights_are_loaded_from_the_named_checkpoint(mod, checkpoint):
    trainer = FakeTrainer()
    mod.load_weights(trainer, str(checkpoint))
    assert trainer.loaded == str(checkpoint)


def test_it_evaluates_exactly_once(mod, checkpoint, tmp_path):
    trainer = FakeTrainer()
    mod.score(trainer, str(checkpoint), {"checkpoint": str(checkpoint)},
              str(tmp_path / "out.json"), log=lambda *a: None)
    assert trainer.evaluated == 1


def test_both_metrics_reach_the_record(mod, checkpoint, tmp_path):
    out = tmp_path / "out.json"
    mod.score(FakeTrainer(), str(checkpoint), {"checkpoint": str(checkpoint)},
              str(out), log=lambda *a: None)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["metrics"]["eval_loss"] == 0.0712
    assert written["metrics"]["eval_loss_mask"] == 0.2673


def test_the_conditions_travel_with_the_number(mod, checkpoint, tmp_path):
    where = mod.conditions(checkpoint=str(checkpoint), eval_batch=32, world_size=4,
                           subset_rows=10000, subset_random=True, subset_seed=42,
                           mask_seed=42, mask_rate=0.2, judged_on="eval_loss")
    out = tmp_path / "out.json"
    mod.score(FakeTrainer(), str(checkpoint), where, str(out), log=lambda *a: None)
    written = json.loads(out.read_text(encoding="utf-8"))
    for key, value in (("per_device_eval_batch_size", 32), ("processes", 4),
                       ("eval_subset_rows", 10000), ("eval_subset_seed", 42),
                       ("eval_mask_seed", 42), ("fixed_eval_mask", True),
                       ("mlm_probability", 0.2), ("judged_on", "eval_loss")):
        assert written[key] == value


def test_the_step_comes_from_the_checkpoint_name(mod):
    assert mod.checkpoint_step("/a/b/checkpoint-32000") == 32000
    assert mod.checkpoint_step("/a/b/checkpoint-32000/") == 32000
    assert mod.checkpoint_step("/a/b/final") is None


def test_the_record_carries_the_step(mod, checkpoint, tmp_path):
    where = mod.conditions(checkpoint=str(checkpoint), eval_batch=32, world_size=4,
                           subset_rows=10000, subset_random=True, subset_seed=42,
                           mask_seed=42, mask_rate=0.2, judged_on="eval_loss")
    assert where["step"] == 32000


def test_only_rank_zero_writes(mod, checkpoint, tmp_path):
    out = tmp_path / "out.json"
    record = mod.score(FakeTrainer(main=False), str(checkpoint),
                       {"checkpoint": str(checkpoint)}, str(out), log=lambda *a: None)
    assert not out.exists()
    # The metrics still came back: every rank evaluates, or the others wait forever.
    assert record["metrics"]["eval_loss"] == 0.0712


def test_a_trainer_that_cannot_load_is_refused_rather_than_scored(mod, checkpoint):
    class NoLoader:
        model = None

        def evaluate(self):
            raise AssertionError("must not evaluate unloaded weights")

    with pytest.raises(SystemExit, match="cannot be shown to have been loaded"):
        mod.load_weights(NoLoader(), str(checkpoint))


def test_the_output_directory_is_created(mod, checkpoint, tmp_path):
    out = tmp_path / "deep" / "deeper" / "out.json"
    mod.score(FakeTrainer(), str(checkpoint), {"checkpoint": str(checkpoint)},
              str(out), log=lambda *a: None)
    assert out.exists()


# --- the hook inside main.py ----------------------------------------------------

def test_main_declares_the_three_names_before_the_configurator(mod):
    text = (repo_root() / "molcrawl" / "models" / "bert" / "main.py").read_text(
        encoding="utf-8")
    head = text.split("config_keys = ")[0]
    for name in ("eval_only = False", 'eval_only_checkpoint = ""',
                 'eval_only_out = ""'):
        assert name in head


def test_main_drops_the_training_callbacks_in_eval_only(mod):
    text = (repo_root() / "molcrawl" / "models" / "bert" / "main.py").read_text(
        encoding="utf-8")
    hook = text.split("if bool(eval_only):")[1]
    assert "callbacks = []" in hook.split("trainer = _WeightDecayNoEmbedTrainer")[0]


def test_main_does_not_train_in_eval_only(mod):
    text = (repo_root() / "molcrawl" / "models" / "bert" / "main.py").read_text(
        encoding="utf-8")
    branch = text.split("_evalonly.score(")[1].split("try:")[0]
    assert "SystemExit(0)" in branch


def test_eval_only_is_off_by_default(mod):
    text = (repo_root() / "molcrawl" / "models" / "bert" / "main.py").read_text(
        encoding="utf-8")
    assert "eval_only = False" in text
    assert "eval_only = True" not in text
