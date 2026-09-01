"""The nanoGPT run manifest has to survive the three failures that motivated it.

Each test here corresponds to a question that could not be answered from the 244
nanoGPT checkpoints on disk: what the effective global batch was, which corpus
was read, and whether a value was chosen or merely inherited.
"""

import json
import os

import pytest

from molcrawl.models.gpt2._run_manifest import MANIFEST, note_resume, write_manifest

DEFAULTS = {
    "learning_rate": 6e-4,
    "batch_size": 12,
    "gradient_accumulation_steps": 40,
    "block_size": 1024,
    "max_iters": 60000,
    "weight_decay": 1e-1,
}


def _write(tmp_path, *, config=None, batch=None, data=None):
    config = {**DEFAULTS, **(config or {})}
    return write_manifest(
        str(tmp_path),
        config,
        DEFAULTS,
        data=data or {"modality": "genome_sequence", "dataset_dir": "/src/gpt2"},
        batch={
            "batch_size": 320,
            "gradient_accumulation_steps_configured": 8,
            "gradient_accumulation_steps_per_rank": 2,
            "world_size": 4,
            "block_size": 1024,
            **(batch or {}),
        },
        schedule={"max_iters": config["max_iters"]},
        objective={"task": "clm"},
        evaluation={"eval_iters": 200},
        selection={"init_from": "scratch"},
        seed={"seed": 1337},
    )


def test_effective_global_batch_is_the_product_before_the_world_size_division(tmp_path):
    """320 x 8 on 4 GPUs is 2,560 -- not 320 x 2 x 4 read the HF way, and not 10,240.

    This is the number that was reported as 2,560 while the run used 640, and the
    only defence is recording the product explicitly rather than its factors.
    """
    manifest = _write(tmp_path)

    assert manifest["batch"]["effective_global_batch"] == 2560
    assert manifest["batch"]["gpu_independent"] is True
    assert manifest["batch"]["tokens_per_step"] == 2560 * 1024
    # The per-rank value is kept, but it is not what the global is derived from.
    assert manifest["batch"]["gradient_accumulation_steps_per_rank"] == 2


def test_a_value_equal_to_the_default_is_recorded_as_default_not_left_silent(tmp_path):
    manifest = _write(tmp_path)
    learning_rate = manifest["sources"]["learning_rate"]

    assert learning_rate["from"] == "default"
    assert learning_rate["value"] == 6e-4
    assert learning_rate["resolved_by"] is None


def test_an_overridden_value_names_the_default_it_replaced(tmp_path):
    """genome ran at 1e-4 against a 6e-4 default and nothing said so."""
    manifest = _write(tmp_path, config={"learning_rate": 1e-4})
    learning_rate = manifest["sources"]["learning_rate"]

    assert learning_rate["from"] == "overridden"
    assert learning_rate["value"] == 1e-4
    assert learning_rate["default"] == 6e-4
    assert "indistinguishable" in learning_rate["resolved_by"]


def test_only_the_nanogpt_values_are_tracked(tmp_path):
    """value_sources itself is covered in test_run_provenance; this is the list."""
    sources = _write(tmp_path)["sources"]

    assert "gradient_accumulation_steps" in sources
    assert "max_iters" in sources
    assert "max_steps" not in sources  # that is the HF name


def test_the_dataset_directory_is_recorded_not_only_the_modality(tmp_path):
    """ckpt.pt says "genome_sequence"; that does not identify a corpus."""
    manifest = _write(
        tmp_path,
        data={
            "modality": "genome_sequence",
            "dataset_dir": "/ls/genome_sequence/mammal_centered/training_ready_hf_dataset_gpt2",
            "rows": {"train": 46_952_621, "eval": 50_002},
        },
    )

    assert manifest["data"]["dataset_dir"].endswith("training_ready_hf_dataset_gpt2")
    assert manifest["data"]["rows"]["train"] == 46_952_621


def test_git_state_reports_whether_the_tree_was_dirty(tmp_path):
    manifest = _write(tmp_path)
    git = manifest["run"]["git"]

    assert set(git) == {"commit", "branch", "dirty", "dirty_files"}
    assert isinstance(git["dirty"], bool)


def test_tracked_environment_variables_are_captured(tmp_path, monkeypatch):
    """The override lives in the environment, which the checkpoint never sees."""
    monkeypatch.setenv("SUBSET_GPT2_LR", "0.0001")
    monkeypatch.setenv("GENOME_SUBSET", "mammal_centered")
    monkeypatch.delenv("GPT2_LR_TAG", raising=False)

    manifest = _write(tmp_path)

    assert manifest["env"]["SUBSET_GPT2_LR"] == "0.0001"
    assert manifest["env"]["GENOME_SUBSET"] == "mammal_centered"
    assert "GPT2_LR_TAG" not in manifest["env"]


def test_manifest_is_written_as_readable_json(tmp_path):
    _write(tmp_path)
    path = os.path.join(str(tmp_path), MANIFEST)

    with open(path) as fh:
        text = fh.read()
    assert text.endswith("\n")
    assert json.loads(text)["framework"] == "nanoGPT"


def test_resume_appends_history_without_losing_the_rest(tmp_path):
    _write(tmp_path)
    note_resume(str(tmp_path), 12000)
    note_resume(str(tmp_path), 24000)

    with open(os.path.join(str(tmp_path), MANIFEST)) as fh:
        manifest = json.load(fh)

    assert manifest["run"]["resumed_from_iter"] == 24000
    assert [h["from_iter"] for h in manifest["run"]["resume_history"]] == [12000, 24000]
    assert manifest["batch"]["effective_global_batch"] == 2560


def test_resume_on_a_missing_manifest_is_a_no_op(tmp_path):
    note_resume(str(tmp_path), 1000)  # must not raise
    assert not os.path.exists(os.path.join(str(tmp_path), MANIFEST))


@pytest.mark.parametrize("missing", [None, 0])
def test_a_missing_batch_size_does_not_crash_the_write(tmp_path, missing):
    manifest = _write(tmp_path, batch={"batch_size": missing})
    assert manifest["batch"]["effective_global_batch"] == 0
    assert manifest["batch"]["tokens_per_step"] == 0
