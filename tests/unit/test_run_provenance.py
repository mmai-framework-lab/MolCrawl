"""The provenance fields both trainers share, and the guarantee that they agree.

The point of the shared module is that a manifest answers the same three
questions whichever trainer wrote it: was the tree clean, was each value chosen
or inherited, and what did the environment set.
"""

import pytest

from molcrawl.models._provenance import (
    COMMON_ENV,
    MAX_DIRTY_FILES,
    dirty_tree_warning,
    environment,
    git_state,
    value_sources,
)

DEFAULTS = {"learning_rate": 6e-4, "max_steps": 60000, "seed": 42}
TRACKED = ("learning_rate", "max_steps", "seed", "not_in_config")


def test_a_value_matching_its_default_is_stated_not_omitted():
    """Silence would leave a reader to assume the agreement instead of reading it."""
    sources = value_sources(DEFAULTS, DEFAULTS, TRACKED)

    assert sources["learning_rate"]["from"] == "default"
    assert sources["learning_rate"]["value"] == 6e-4
    assert sources["learning_rate"]["resolved_by"] is None


def test_an_overridden_value_names_the_default_it_replaced():
    """genome ran at 1e-4 against a 6e-4 default and nothing recorded the pair."""
    sources = value_sources({**DEFAULTS, "learning_rate": 1e-4}, DEFAULTS, TRACKED)

    assert sources["learning_rate"] == {
        "value": 1e-4,
        "from": "overridden",
        "default": 6e-4,
        "resolved_by": "config file, env var or --key=value (indistinguishable)",
    }


def test_the_three_override_paths_are_not_claimed_to_be_distinguishable():
    """A config file, an env var and --key=value all overwrite the same globals."""
    sources = value_sources({"max_steps": 1}, DEFAULTS, TRACKED)
    assert "indistinguishable" in sources["max_steps"]["resolved_by"]


def test_a_key_the_config_never_carried_is_skipped():
    assert "not_in_config" not in value_sources(DEFAULTS, DEFAULTS, TRACKED)


def test_a_key_with_no_default_is_still_reported_as_overridden():
    """Names declared only by a config file have no module default to match."""
    sources = value_sources({"judge_on": "eval_loss_mask"}, {}, ("judge_on",))

    assert sources["judge_on"]["from"] == "overridden"
    assert sources["judge_on"]["default"] is None


def test_git_state_answers_whether_the_tree_was_clean():
    state = git_state()

    assert set(state) == {"commit", "branch", "dirty", "dirty_files"}
    assert isinstance(state["dirty"], bool)
    assert len(state["dirty_files"]) <= MAX_DIRTY_FILES


def test_a_clean_tree_produces_no_warning():
    assert dirty_tree_warning({"dirty": False, "commit": "abc1234", "dirty_files": []}) is None


def test_a_dirty_tree_warning_says_the_run_is_not_reproducible_from_the_commit():
    warning = dirty_tree_warning(
        {"dirty": True, "commit": "abc1234", "dirty_files": ["a.py", "b.py"]}
    )

    assert "NOT reproducible" in warning
    assert "abc1234" in warning
    assert "2 file(s)" in warning


def test_the_warning_survives_a_state_missing_its_file_list():
    assert dirty_tree_warning({"dirty": True, "commit": None}) is not None


def test_environment_captures_the_variables_that_are_set(monkeypatch):
    monkeypatch.setenv("LEARNING_SOURCE_DIR", "/ls")
    monkeypatch.setenv("SUBSET_BERT_EPOCHS", "9")
    monkeypatch.delenv("GENOME_SUBSET", raising=False)

    captured = environment(("SUBSET_BERT_EPOCHS",))

    assert captured["LEARNING_SOURCE_DIR"] == "/ls"
    assert captured["SUBSET_BERT_EPOCHS"] == "9"


def test_environment_omits_unset_names_rather_than_recording_null(monkeypatch):
    """A list of nulls says what could have shaped the run, not what did."""
    for name in COMMON_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SUBSET_GPT2_LR", raising=False)

    assert environment(("SUBSET_GPT2_LR",)) == {}


@pytest.mark.parametrize(
    "module",
    ["molcrawl.models.gpt2._run_manifest", "molcrawl.models.bert._run_manifest"],
)
def test_both_trainers_expose_the_same_provenance_surface(module):
    """If one grows a field the other lacks, the manifests stop being comparable."""
    import importlib

    manifest = importlib.import_module(module)

    assert manifest.TRACKED, f"{module} tracks no values"
    assert manifest.TRACKED_ENV, f"{module} tracks no environment variables"
    assert callable(manifest.dirty_tree_warning)
    # COMMON_ENV is added by environment(); duplicating it per trainer would let
    # the two drift apart.
    assert not set(manifest.TRACKED_ENV) & set(COMMON_ENV)
