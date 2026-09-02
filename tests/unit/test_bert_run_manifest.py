"""#154 wrote what the run was; these are the three things it did not write.

Each corresponds to a value that was resolved somewhere the BERT run never
recorded: the epoch count the schedule is derived from, whether a value was
chosen or inherited, and whether the checkout was clean.
"""

import json
import os

from molcrawl.models.bert._run_manifest import MANIFEST, note_resume, write_manifest


class Args:
    """The subset of TrainingArguments the manifest reads."""

    per_device_train_batch_size = 160
    gradient_accumulation_steps = 4
    world_size = 4
    max_steps = 111230
    warmup_steps = 2224
    lr_scheduler_type = "linear"
    learning_rate = 1e-4
    metric_for_best_model = "eval_loss_mask"
    save_steps = 1000
    eval_steps = 1000
    save_total_limit = None
    seed = 42
    data_seed = None


DEFAULTS = {"learning_rate": 5e-5, "max_steps": 0, "seed": 42, "max_length": 512}


def _write(tmp_path, *, resolved=None, defaults=DEFAULTS):
    return write_manifest(
        str(tmp_path),
        Args(),
        config={"max_steps_source": "config-derived", "epochs_planned": 3},
        data={"dataset_dir": "/ls/genome/mammal_centered/training_ready_hf_dataset_bert"},
        objective={"task": "mlm", "seq_len": 512, "mlm_probability": 0.2},
        evaluation={"judge_on": "eval_loss_mask", "seq_len": 512},
        resolved=resolved if resolved is not None else dict(DEFAULTS),
        defaults=defaults,
    )


def test_effective_global_batch_multiplies_by_the_world_size(tmp_path):
    """HF semantics, unlike nanoGPT: 160 x 4 x 4 GPUs is 2,560."""
    manifest = _write(tmp_path)

    assert manifest["batch"]["effective_global_batch"] == 2560
    assert manifest["batch"]["world_size"] == 4


def test_a_value_matching_its_default_is_recorded_as_default(tmp_path):
    sources = _write(tmp_path)["sources"]

    assert sources["seed"]["from"] == "default"
    assert sources["seed"]["value"] == 42


def test_an_overridden_learning_rate_names_the_default_it_replaced(tmp_path):
    sources = _write(tmp_path, resolved={**DEFAULTS, "learning_rate": 1e-4})["sources"]

    assert sources["learning_rate"]["from"] == "overridden"
    assert sources["learning_rate"]["value"] == 1e-4
    assert sources["learning_rate"]["default"] == 5e-5


def test_the_epoch_environment_variable_is_captured(tmp_path, monkeypatch):
    """genome derives max_steps from SUBSET_BERT_EPOCHS; the 9-epoch run set it
    and the run said so nowhere."""
    monkeypatch.setenv("SUBSET_BERT_EPOCHS", "9")
    monkeypatch.setenv("GENOME_SUBSET", "mammal_centered")

    env = _write(tmp_path)["env"]

    assert env["SUBSET_BERT_EPOCHS"] == "9"
    assert env["GENOME_SUBSET"] == "mammal_centered"


def test_git_state_replaces_the_bare_commit(tmp_path):
    """#154 recorded only the hash, which does not say the run came from it."""
    run = _write(tmp_path)["run"]

    assert set(run["git"]) == {"commit", "branch", "dirty", "dirty_files"}
    # The hash moved under git rather than being duplicated beside it.
    assert "commit" not in run


def test_the_curated_config_still_reaches_the_schedule(tmp_path):
    """resolved/defaults are additions; they must not displace what #154 wrote."""
    schedule = _write(tmp_path)["schedule"]

    assert schedule["max_steps"] == 111230
    assert schedule["epochs_planned"] == 3
    assert schedule["max_steps_source"] == "config-derived"


def test_selection_records_judge_on_beside_the_hf_metric(tmp_path):
    selection = _write(tmp_path)["selection"]

    assert selection["metric_for_best_model"] == "eval_loss_mask"
    assert selection["judge_on"] == "eval_loss_mask"


def test_missing_resolved_and_defaults_do_not_break_the_write(tmp_path):
    """Bookkeeping never stops a run, so absent provenance degrades to empty."""
    manifest = write_manifest(
        str(tmp_path),
        Args(),
        config={},
        data={},
        objective={"seq_len": 512},
        evaluation={},
    )

    assert manifest["sources"] == {}
    assert manifest["batch"]["effective_global_batch"] == 2560


def test_resume_appends_history_without_losing_the_rest(tmp_path):
    _write(tmp_path)
    note_resume(str(tmp_path), 60000)

    with open(os.path.join(str(tmp_path), MANIFEST)) as fh:
        manifest = json.load(fh)

    assert manifest["run"]["resumed_from_step"] == 60000
    assert manifest["batch"]["effective_global_batch"] == 2560
    assert manifest["sources"]["seed"]["from"] == "default"


def _batch(tmp_path, *, pd, ga, ws, config):
    class _A(Args):
        per_device_train_batch_size = pd
        gradient_accumulation_steps = ga
        world_size = ws

    return write_manifest(
        str(tmp_path), _A(), config=config, data={},
        objective={"task": "mlm", "seq_len": 1024}, evaluation={"seq_len": 1024},
    )["batch"]


def test_the_declared_batch_and_the_verdict_are_both_recorded(tmp_path):
    """A manifest carrying only the actual batch cannot say a check ran."""
    got = _batch(tmp_path, pd=8, ga=80, ws=4, config={"expected_global_batch": 2560})

    assert got["effective_global_batch"] == 2560
    assert got["expected_global_batch"] == 2560
    assert got["matches_expected_global_batch"] is True


def test_a_deliberate_deviation_is_recorded_rather_than_waved_through(tmp_path):
    """--expected_global_batch=640 on a one-GPU smoke run has to leave a trace."""
    got = _batch(tmp_path, pd=8, ga=80, ws=1, config={"expected_global_batch": 640})

    assert got["expected_global_batch"] == 640
    assert got["effective_global_batch"] == 640


def test_no_declaration_is_distinguishable_from_a_check_that_passed(tmp_path):
    """None must not read as agreement; no check was made at all."""
    got = _batch(tmp_path, pd=8, ga=80, ws=4, config={})

    assert got["expected_global_batch"] is None
    assert got["matches_expected_global_batch"] is None


def test_a_decomposition_change_that_preserves_the_product_still_matches(tmp_path):
    """160x4 and 8x80 are the same run to declare; only throughput differs."""
    got = _batch(tmp_path, pd=160, ga=4, ws=4, config={"expected_global_batch": 2560})

    assert got["matches_expected_global_batch"] is True
