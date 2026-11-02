"""The end of a learning rate schedule, read from a config and from a run.

The table this covers exists because a claim about every modality decaying to zero
was made without being checked, so the checks here are about the two ways such a
read goes wrong: taking the last line of a log for the end of the schedule, and
failing to follow a two-line override config back to the values it inherits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to.

    Not git's common dir: from a worktree that resolves to the main checkout, and
    the script under test is the worktree's copy.
    """
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    import importlib.util

    path = repo_root() / "scripts" / "lr_schedule_end_table.py"
    spec = importlib.util.spec_from_file_location("lr_schedule_end_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(tmp_path: Path, rates, global_step) -> Path:
    """A run directory holding one checkpoint with a log_history."""
    run = tmp_path / "a-run"
    checkpoint = run / f"checkpoint-{global_step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": global_step,
        "log_history": [{"step": s, "learning_rate": r} for s, r in rates],
    }))
    return run


def write_configs(tmp_path: Path, files: dict) -> Path:
    """A fake configs package, so a star import has somewhere to resolve to."""
    base = tmp_path / "molcrawl" / "tasks" / "pretrain" / "configs" / "m"
    base.mkdir(parents=True)
    for name, text in files.items():
        (base / name).write_text(text)
    return tmp_path


# --- the endpoint is the schedule's, not the log's last line ------------------

def test_linear_ends_at_zero_even_when_the_last_logged_rate_is_not(mod, tmp_path):
    """The real case: protein, rna and genome all log 1e-7 on their last line."""
    repo = write_configs(tmp_path, {"c.py": "learning_rate = 3e-4\nmax_steps = 1000\n"})
    run = write_run(tmp_path, [(100, 3e-4), (900, 3.08e-7)], 1000)
    row = mod.bert_row("protein_sequence", run, repo / "molcrawl/tasks/pretrain/configs/m/c.py", repo)
    assert row["ends_at_zero"] == "yes"
    assert row["schedule_end_lr"] == "0.000e+00"
    assert row["last_logged_lr"] == "3.080e-07"
    assert row["last_logged_step"] == 900


def test_constant_with_warmup_does_not_end_at_zero(mod, tmp_path):
    """compounds' extension stages: the rate holds a tenth of the stage-1 peak."""
    repo = write_configs(tmp_path, {
        "c.py": 'learning_rate = 1e-4\nmax_steps = 45000\n'
                'lr_scheduler_type = "constant_with_warmup"\n'})
    run = write_run(tmp_path, [(100, 6.6e-5), (45000, 1e-4)], 45000)
    row = mod.bert_row("compounds", run, repo / "molcrawl/tasks/pretrain/configs/m/c.py", repo)
    assert row["ends_at_zero"] == "no"
    assert row["schedule_end_lr"] == "1.000e-04"
    assert row["schedule"] == "constant_with_warmup"


def test_a_zero_in_the_middle_of_the_history_is_not_the_end(mod, tmp_path):
    """log_history is restored on resume, so stage 1's zero is still in the list."""
    repo = write_configs(tmp_path, {
        "c.py": 'learning_rate = 1e-4\nlr_scheduler_type = "constant_with_warmup"\n'})
    run = write_run(tmp_path, [(15000, 0.0), (15500, 1e-4), (45000, 1e-4)], 45000)
    row = mod.bert_row("compounds", run, repo / "molcrawl/tasks/pretrain/configs/m/c.py", repo)
    assert row["ends_at_zero"] == "no"
    assert row["last_logged_lr"] == "1.000e-04"


def test_a_run_with_no_checkpoint_says_so_instead_of_guessing(mod, tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    row = mod.bert_row("rna", run, None, tmp_path)
    assert "no checkpoint" in row["schedule"]
    assert row.get("ends_at_zero", "") == ""


# --- inheritance and expressions ---------------------------------------------

def test_a_two_line_override_reports_the_inherited_schedule(mod, tmp_path):
    """molnl's fixedval config sets only the rate; the length comes from its parent."""
    repo = write_configs(tmp_path, {
        "base.py": "max_iters = 1500\nlr_decay_iters = 1500\nwarmup_iters = 30\n"
                   "learning_rate = 6e-4\nmin_lr = 6e-5\n",
        "child.py": "from molcrawl.tasks.pretrain.configs.m.base import *\n"
                    "learning_rate = 1.2e-3\nmin_lr = 1.2e-3 / 10\n",
    })
    config = repo / "molcrawl/tasks/pretrain/configs/m/child.py"
    run = tmp_path / "gpt2-run"
    run.mkdir()
    row = mod.gpt2_row("molecule_nat_lang", run, config, repo)
    assert row["peak_lr"] == "1.200e-03"          # the child's
    assert row["schedule_end_lr"] == "1.200e-04"  # 1.2e-3 / 10, evaluated
    assert "lr_decay_iters=1500" in row["schedule"]  # the parent's
    assert row["ends_at_zero"] == "no"


def test_min_lr_written_against_learning_rate_resolves(mod, tmp_path):
    """genome's subset config writes min_lr = learning_rate / 10."""
    repo = write_configs(tmp_path, {
        "c.py": 'learning_rate = float(os.environ.get("SUBSET_GPT2_LR", "0.0006"))\n'
                "min_lr = learning_rate / 10\nlr_decay_iters = 1000\n",
    })
    run = tmp_path / "gpt2-run"
    run.mkdir()
    row = mod.gpt2_row("genome_sequence", run, repo / "molcrawl/tasks/pretrain/configs/m/c.py", repo)
    assert row["peak_lr"] == "6.000e-04"
    assert row["schedule_end_lr"] == "6.000e-05"


def test_a_min_lr_of_zero_is_reported_as_reaching_zero(mod, tmp_path):
    """No config does this today; the column has to be able to say it if one did."""
    repo = write_configs(tmp_path, {
        "c.py": "learning_rate = 6e-4\nmin_lr = 0.0\nlr_decay_iters = 1000\n"})
    run = tmp_path / "gpt2-run"
    run.mkdir()
    row = mod.gpt2_row("rna", run, repo / "molcrawl/tasks/pretrain/configs/m/c.py", repo)
    assert row["ends_at_zero"] == "yes"


def test_a_circular_star_import_does_not_recurse_forever(mod, tmp_path):
    repo = write_configs(tmp_path, {
        "a.py": "from molcrawl.tasks.pretrain.configs.m.b import *\nlearning_rate = 1e-4\n",
        "b.py": "from molcrawl.tasks.pretrain.configs.m.a import *\nmin_lr = 1e-5\n",
    })
    values = mod.config_values(repo / "molcrawl/tasks/pretrain/configs/m/a.py", repo)
    assert values["learning_rate"][0] == "1e-4"
    assert values["min_lr"][0] == "1e-5"


def test_the_nanogpt_step_count_comes_from_the_run_not_the_config(mod, tmp_path):
    """max_iters is derived from a dataset row count in the subset configs."""
    repo = write_configs(tmp_path, {
        "c.py": "learning_rate = 6e-4\nmin_lr = 6e-5\n"
                "max_iters = _iters_for(_N_EPOCH)\nlr_decay_iters = max_iters\n"})
    run = tmp_path / "gpt2-run"
    run.mkdir()
    (run / "logging_20261102_000000.csv").write_text(
        "iter,train_loss,val_loss\n100,1.0,1.1\n105000,0.9,1.0\n")
    row = mod.gpt2_row("genome_sequence", run, repo / "molcrawl/tasks/pretrain/configs/m/c.py", repo)
    assert row["steps_run"] == "105000"


def test_an_expression_naming_something_unknown_returns_none(mod):
    assert mod.as_number("some_other_name / 10") is None
    assert mod.as_number("1.2e-3 / 10") == pytest.approx(1.2e-4)
