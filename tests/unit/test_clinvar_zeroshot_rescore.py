"""Which checkpoint the zero-shot rescorer scores.

The runs moved between checkouts after they finished, so the absolute path in
``best_model_checkpoint`` often does not exist any more. The step in its
basename still does, and the directory is looked for under the run.

What this pins down is the other half: when the adopted checkpoint cannot be
resolved, the rescorer stops. It used to score the largest step instead, and
the output said ``checkpoint`` without saying that the checkpoint was the final
one rather than the adopted one.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "clinvar_zeroshot_bert_rescore.py"
_spec = importlib.util.spec_from_file_location("_clinvar_rescore", _SRC)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def _run(tmp_path, steps, best=None, state_extra=None):
    for step in steps:
        (tmp_path / f"checkpoint-{step}").mkdir()
    state = dict(state_extra or {})
    if best is not None:
        state["best_model_checkpoint"] = best
    (tmp_path / f"checkpoint-{max(steps)}" / "trainer_state.json").write_text(
        json.dumps(state))
    return str(tmp_path)


def test_the_adopted_step_comes_back_with_the_path(tmp_path):
    run = _run(tmp_path, (1000, 2000, 3000), best=str(tmp_path / "checkpoint-1000"))

    path, step = rs.adopted_checkpoint(run)

    assert path.endswith("checkpoint-1000")    # not the newest, which is 3000
    assert step == 1000


def test_a_path_from_another_checkout_resolves_by_its_step(tmp_path):
    """The field holds an absolute path into the worktree the run came from."""
    run = _run(tmp_path, (1000, 2000),
               best="/gone/MolCrawl-wt-genome/learning_source/run/checkpoint-1000")

    path, step = rs.adopted_checkpoint(run)

    assert path == str(tmp_path / "checkpoint-1000")
    assert step == 1000


def test_an_unnamed_adopted_checkpoint_stops_instead_of_taking_the_last(tmp_path):
    run = _run(tmp_path, (1000, 2000), state_extra={"log_history": []})

    with pytest.raises(SystemExit) as caught:
        rs.adopted_checkpoint(run)

    assert "best_model_checkpoint" in str(caught.value)
    assert "2000" in str(caught.value)


def test_a_named_checkpoint_that_is_not_on_disk_stops(tmp_path):
    run = _run(tmp_path, (2000,), best=str(tmp_path / "checkpoint-1000"))

    with pytest.raises(SystemExit) as caught:
        rs.adopted_checkpoint(run)

    assert "checkpoint-1000" in str(caught.value)


def test_a_run_with_no_checkpoint_stops(tmp_path):
    with pytest.raises(SystemExit):
        rs.adopted_checkpoint(str(tmp_path))
