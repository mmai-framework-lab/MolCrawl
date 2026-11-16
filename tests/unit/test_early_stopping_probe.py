"""The rules the early-stopping probe replays, and the expected steps it writes down.

The probe's verdict is only as good as these two replays: each must stop where the
code it stands for stops. The cases below are the probe's own cases (constant loss)
plus the shapes that tell the two rules apart.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    path = repo_root() / "scripts" / "early_stopping_probe.py"
    spec = importlib.util.spec_from_file_location("early_stopping_probe", path)
    m = importlib.util.module_from_spec(spec)
    # dataclass resolves its string annotations through sys.modules[cls.__module__],
    # so the module has to be registered before it executes.
    sys.modules[spec.name] = m
    try:
        spec.loader.exec_module(m)
        yield m
    finally:
        sys.modules.pop(spec.name, None)


def flat(steps, value=2.0):
    return [(s, value) for s in steps]


# --------------------------------------------------------------------------- GPT-2


def test_gpt2_constant_loss_stops_patience_evals_after_the_first(mod):
    stop, counter, best = mod.gpt2_replay(flat(range(0, 101, 10)), patience=3)
    assert (stop, counter, best) == (30, 3, 2.0)


def test_gpt2_equal_is_not_an_improvement(mod):
    evals = [(0, 2.0), (10, 1.9), (20, 1.9), (30, 1.9), (40, 1.9)]
    assert mod.gpt2_replay(evals, patience=3)[0] == 40


def test_gpt2_any_strict_improvement_resets(mod):
    evals = [(0, 2.0), (10, 2.0), (20, 2.0), (30, 1.99999), (40, 2.0), (50, 2.0)]
    stop, counter, _ = mod.gpt2_replay(evals, patience=3)
    assert stop is None and counter == 2


def test_gpt2_resume_carries_the_counter_and_counts_the_resume_step_again(mod):
    _, counter, best = mod.gpt2_replay(flat([0, 10, 20]), patience=3)
    assert counter == 2
    stop, _, _ = mod.gpt2_replay(flat([20, 30, 40]), patience=3, best=best, counter=counter)
    assert stop == 20


# --------------------------------------------------------------------------- HF


def test_hf_constant_loss_saving_every_eval(mod):
    stop, counter, best, _ = mod.hf_replay(flat(range(10, 101, 10)), 3, save_steps=10)
    assert (stop, counter, best) == (40, 3, 2.0)


def test_hf_best_metric_waits_for_a_save(mod):
    # save every 20: evals at 10 and 20 both see best_metric None
    assert mod.hf_replay(flat(range(10, 101, 10)), 3, save_steps=20)[0] == 50


def test_hf_save_on_improve_sets_best_metric_at_the_first_eval(mod):
    assert mod.hf_replay(flat(range(10, 101, 10)), 3, save_steps=20, save_on_improve=True)[0] == 40


def test_hf_resume_with_the_counter_reset(mod):
    stop, counter, best, _ = mod.hf_replay(flat([10, 20, 30]), 3, save_steps=10)
    assert stop is None and counter == 2
    assert mod.hf_replay(flat(range(40, 101, 10)), 3, save_steps=10, best_metric=best)[0] == 60


def test_hf_improvement_between_saves_is_judged_against_the_stale_best(mod):
    # 1.5 at step 30 is not saved (save every 20); 40 beats the stored 2.0 -> reset.
    evals = [(10, 2.0), (20, 2.0), (30, 1.5), (40, 1.8), (50, 1.8), (60, 1.8), (70, 1.8)]
    stop, *_ = mod.hf_replay(evals, 3, save_steps=20)
    # 30: 1.5<2.0 reset; 40: 1.8<2.0 reset, save -> best 1.8; 50,60,70 -> stop at 70
    assert stop == 70
    assert mod.gpt2_replay([(0, 9)] + evals, 3)[0] == 60


# --------------------------------------------------------------------------- cases


def test_every_expected_step_is_what_the_replay_gives_on_constant_loss(mod):
    for case in mod.CASES:
        if case.arch == "gpt2":
            best, counter, stop = 1e9, 0, None
            for i, ph in enumerate(case.phases):
                start = 0 if i == 0 else case.phases[i - 1].overrides["max_iters"]
                ev = flat(range(start, ph.overrides["max_iters"] + 1, mod.EVAL_EVERY))
                stop, counter, best = mod.gpt2_replay(ev, mod.PATIENCE, best=best, counter=counter)
                if stop is not None:
                    break
            end = stop if stop is not None else case.phases[-1].overrides["max_iters"]
        else:
            best, stop = None, None
            save = case.phases[-1].overrides.get("save_steps", mod.BERT_COMMON["save_steps"])
            soi = case.phases[-1].overrides.get("save_on_improve", False)
            for i, ph in enumerate(case.phases):
                start = 0 if i == 0 else case.phases[i - 1].overrides["max_steps"]
                ev = flat(range(start + mod.EVAL_EVERY, ph.overrides["max_steps"] + 1, mod.EVAL_EVERY))
                stop, _, best, _ = mod.hf_replay(ev, mod.PATIENCE, save, soi, best_metric=best)
                if stop is not None:
                    break
            end = stop if stop is not None else case.phases[-1].overrides["max_steps"]
        assert end == case.expected, case.name


# --------------------------------------------------------------------------- reading


def test_read_bert_evals_counts_from_the_resume_point(mod):
    log = "\n".join([
        "   Resuming training from: /x/runs/b_resume/checkpoint-30",
        "{'loss': 7.1, 'grad_norm': 1.0, 'learning_rate': 0.0, 'epoch': 0.0}",
        "{'eval_loss': 5.0, 'eval_loss_mask': 5.25, 'eval_runtime': 3.1, 'epoch': 0.0}",
        "{'loss': 7.1, 'grad_norm': 1.0, 'learning_rate': 0.0, 'epoch': 0.0}",
        "{'eval_loss': 5.0, 'eval_loss_mask': 5.25, 'eval_runtime': 3.0, 'epoch': 0.0}",
    ])
    start, evals = mod.read_bert_evals(log)
    assert start == 30 and evals == [(40, 5.25), (50, 5.25)]


def test_read_gpt2_evals_one_list_per_launch(mod, tmp_path):
    (tmp_path / "logging_20261116_100000.csv").write_text(
        "iter, train_loss, val_loss\n0, 5.0, 5.1\n10, 5.0, 5.1\n20, 5.0, 5.1\n")
    (tmp_path / "logging_20261116_100500.csv").write_text("iter, train_loss, val_loss\n20, 5.0, 5.1\n")
    (tmp_path / "logging_20261116_100501.csv").write_text("iter, train_loss, val_loss\n")
    assert mod.read_gpt2_evals(str(tmp_path)) == [[(0, 5.1), (10, 5.1), (20, 5.1)], [(20, 5.1)]]


def test_manifest_flag_is_found_where_train_py_writes_it(mod, tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"config": {"selection": {"early_stopping": True, "init_from": "scratch"}}}))
    assert mod.gpt2_manifest_flag(str(tmp_path)) is True


def test_configs_import_the_base_and_pin_seed_42(mod, tmp_path):
    mod.write_configs(str(tmp_path))
    plan = (tmp_path / "plan.tsv").read_text().splitlines()
    assert len(plan) == sum(len(c.phases) for c in mod.CASES)
    for row in plan:
        _, _, arch, cfg, run_dir = row.split("\t")
        text = Path(cfg).read_text()
        base = mod.GPT2_BASE if arch == "gpt2" else mod.BERT_BASE
        assert f"from {base[:-3].replace('/', '.')} import *" in text
        assert "\nseed = 42\n" in text
        assert repr(run_dir) in text
        assert (repo_root() / base).exists()
