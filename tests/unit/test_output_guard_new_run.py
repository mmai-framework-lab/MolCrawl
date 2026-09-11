"""The nanoGPT output check applies to runs that start, not to runs that resume."""

import pytest

from molcrawl.core.output_guard import assert_output_dir_for_new_run


@pytest.fixture
def input_tree(tmp_path, monkeypatch):
    src = tmp_path / "learning_source"
    (src / "protein_sequence").mkdir(parents=True)
    monkeypatch.setenv("LEARNING_SOURCE_DIR", str(src))
    for name in ("GENOME_SOURCE_ROOT", "SRC_ROOT"):
        monkeypatch.delenv(name, raising=False)
    return src


def test_a_resume_inside_the_input_tree_is_let_through(input_tree):
    """Refusing it could not move it, only kill it (protein retrain21 at its restart)."""
    assert_output_dir_for_new_run(input_tree / "protein_sequence" / "runs" / "x", "resume")


@pytest.mark.parametrize("init_from", ["scratch", "gpt2", "gpt2-medium", "gpt2-xl"])
def test_every_way_of_starting_a_run_inside_the_input_tree_is_refused(input_tree, init_from):
    """gpt2* initialisations start a new run in a new place, exactly as scratch does."""
    with pytest.raises(ValueError, match="inside an input tree"):
        assert_output_dir_for_new_run(input_tree / "protein_sequence" / "gpt2-output", init_from)


def test_a_new_run_outside_the_input_tree_passes(input_tree, tmp_path):
    assert_output_dir_for_new_run(tmp_path / "runs" / "gpt2-small", "scratch")
