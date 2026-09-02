"""The provenance fields both trainers share, and the guarantee that they agree.

The point of the shared module is that a manifest answers the same three
questions whichever trainer wrote it: was the tree clean, was each value chosen
or inherited, and what did the environment set.
"""

import pytest

from molcrawl.models._provenance import (
    COMMON_ENV,
    MAX_DIRTY_FILES,
    configurator_locals,
    dirty_tree_warning,
    environment,
    git_state,
    introduced_values,
    placement,
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


def test_placement_is_none_not_false_when_there_is_no_allocation(monkeypatch):
    """Outside SLURM nothing disagrees; False would warn on every local run."""
    for name in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES", "SLURM_GPUS",
                 "SLURM_GPUS_ON_NODE", "SLURM_JOB_NODELIST"):
        monkeypatch.delenv(name, raising=False)

    assert placement(1)["world_size_matches_allocation"] is None


def test_placement_catches_a_world_smaller_than_the_allocation(monkeypatch):
    """4 processes against 8 allocated GPUs: the case this exists to name."""
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURM_GPUS", "8")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")

    got = placement(4)
    assert got["gpus_allocated"] == 8
    assert got["world_size_matches_allocation"] is False


def test_placement_infers_the_total_when_only_the_per_node_count_is_set(monkeypatch):
    """--gres forms leave SLURM_GPUS unset, so the total has to be reconstructed."""
    monkeypatch.delenv("SLURM_GPUS", raising=False)
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")

    assert placement(8)["world_size_matches_allocation"] is True


@pytest.mark.parametrize(
    "module",
    ["molcrawl.models.gpt2._run_manifest", "molcrawl.models.bert._run_manifest"],
)
def test_both_trainers_record_the_placement_they_got(module):
    """One trainer knowing what it ran on and the other not is the drift this prevents."""
    import importlib

    assert callable(importlib.import_module(module)._placement)


# --- introduced_values: what value_sources structurally cannot see -----------


def test_a_name_the_config_added_is_reported():
    """eos_token_id is set by 47 configs and no trainer declares it."""
    before = ["learning_rate", "max_steps"]
    after = {"learning_rate": 1e-4, "max_steps": 100, "eos_token_id": 0}

    assert introduced_values(before, after) == {"eos_token_id": 0}


def test_a_name_the_config_merely_overrode_is_not_reported():
    """Those belong to sources, which can say what they replaced."""
    assert introduced_values(["learning_rate"], {"learning_rate": 1e-4}) == {}


def test_names_already_reported_elsewhere_are_not_excluded():
    """An exclusion list is one more list to forget to update (cf. TRACKED)."""
    result = introduced_values([], {"expected_global_batch": 2560, "dataset_dir": "/ls"})

    assert result == {"dataset_dir": "/ls", "expected_global_batch": 2560}


def test_the_configurators_own_locals_are_dropped(tmp_path):
    """exec runs it at module level, so `arg` and `key` become globals too."""
    configurator = tmp_path / "configurator.py"
    configurator.write_text(
        "import sys\n"
        "for arg in sys.argv[1:]:\n"
        "    config_file = arg\n"
        "    key, val = arg.split('=')\n"
        "    try:\n"
        "        attempt = 1\n"
        "    except ValueError:\n"
        "        attempt = 2\n"
    )
    after = {
        "arg": "--lr=1", "key": "lr", "val": "1", "config_file": "c.py",
        "attempt": 1, "eos_token_id": 0,
    }

    assert introduced_values([], after, str(configurator)) == {"eos_token_id": 0}


def test_the_real_configurator_binds_the_names_we_expect():
    """Read from source, so the set cannot drift out of step with the file."""
    names = configurator_locals("molcrawl/models/gpt2/configurator.py")

    assert {"arg", "config_file", "key", "val", "attempt"} <= names


def test_an_unreadable_configurator_excludes_nothing_rather_than_failing():
    assert configurator_locals("/nonexistent/configurator.py") == frozenset()


def test_the_result_is_ordered_so_manifests_diff_cleanly():
    assert list(introduced_values([], {"z": 1, "a": 2, "m": 3})) == ["a", "m", "z"]
