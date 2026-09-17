"""RunLifecycle, launcher_record, and the GPT-2 manifest's schema-2 sections."""

import json
import os

import pytest

from molcrawl.models import _provenance as p
from molcrawl.models.gpt2 import _run_manifest as gm


def _read(path):
    with open(path) as fh:
        return json.load(fh)


def test_starting_running_completed(tmp_path):
    path = str(tmp_path / "run_manifest.json")
    life = p.RunLifecycle(path, "local-1", enabled=True)
    life.start({"run": {"job_id": None}})
    assert _read(path)["run"]["status"] == "starting"

    life.running({"run": {"job_id": None}, "batch": {"effective_global_batch": 2560}})
    doc = _read(path)
    assert doc["run"]["status"] == "running"
    assert doc["run"]["run_id"] == "local-1"
    assert [h["status"] for h in doc["run"]["status_history"]] == ["starting", "running"]

    life.complete(reached_step=10, stop_reason="max_iters")
    doc = _read(path)
    assert doc["run"]["status"] == "completed"
    assert doc["run"]["exit_code"] == 0
    assert doc["run"]["reached_step"] == 10
    assert doc["run"]["end_time"] and doc["run"]["wall_clock_seconds"] >= 0
    assert [h["status"] for h in doc["run"]["status_history"]] == ["starting", "running", "completed"]


def test_an_exception_on_the_way_out_is_recorded_as_failed(tmp_path):
    path = str(tmp_path / "run_manifest.json")
    life = p.RunLifecycle(path, "local-2", enabled=True)
    life.start({"run": {}})
    life._step_getter = lambda: 7
    life._failure_fields = lambda: {"last_checkpoint": "/x/checkpoint-5"}
    life._exc = RuntimeError("CUDA out of memory")
    life._at_exit()
    doc = _read(path)
    assert doc["run"]["status"] == "failed"
    assert doc["run"]["failure"] == {"phase": "setup", "error_summary": "RuntimeError: CUDA out of memory"}
    assert doc["run"]["reached_step"] == 7
    assert doc["run"]["last_checkpoint"] == "/x/checkpoint-5"
    assert doc["run"]["exit_code"] is None


def test_an_exit_without_completion_is_failed_not_left_running(tmp_path):
    path = str(tmp_path / "run_manifest.json")
    life = p.RunLifecycle(path, "local-3", enabled=True)
    life.start({"run": {}})
    life.running({"run": {}})
    life._at_exit()
    doc = _read(path)
    assert doc["run"]["status"] == "failed"
    assert doc["run"]["failure"]["phase"] == "training"
    assert "before reporting completion" in doc["run"]["failure"]["error_summary"]


def test_a_completed_run_is_not_rewritten_at_exit(tmp_path):
    path = str(tmp_path / "run_manifest.json")
    life = p.RunLifecycle(path, "local-4", enabled=True)
    life.start({"run": {}})
    life.complete(reached_step=1)
    life._at_exit()
    assert _read(path)["run"]["status"] == "completed"


def test_disabled_lifecycle_writes_nothing(tmp_path):
    path = str(tmp_path / "run_manifest.json")
    life = p.RunLifecycle(path, "rank-3", enabled=False)
    life.start({"run": {}})
    life.running({"run": {}})
    life.complete()
    assert not os.path.exists(path)


def test_a_resumed_segment_keeps_the_previous_one(tmp_path):
    path = str(tmp_path / "run_manifest.json")
    first = p.RunLifecycle(path, "job1-a", enabled=True)
    first.start({"run": {"job_id": "1"}})
    first.running({"run": {"job_id": "1"}, "placement": {"world_size": 4}, "batch": {"effective_global_batch": 2560}})
    first.complete(reached_step=100)

    second = p.RunLifecycle(path, "job2-b", enabled=True)
    second.start({"run": {"job_id": "2"}}, resume_expected=True)
    doc = _read(path)
    assert doc["run"]["run_id"] == "job2-b"
    seg = doc["run"]["segments"]
    assert len(seg) == 1
    assert seg[0]["run_id"] == "job1-a" and seg[0]["status"] == "completed"
    assert seg[0]["reached_step"] == 100 and seg[0]["world_size"] == 4
    assert seg[0]["effective_global_batch"] == 2560

    second.running({"run": {"job_id": "2"}})
    assert _read(path)["run"]["segments"] == seg


def test_a_fresh_run_does_not_adopt_an_old_manifest(tmp_path):
    path = str(tmp_path / "run_manifest.json")
    old = p.RunLifecycle(path, "old", enabled=True)
    old.start({"run": {}})
    new = p.RunLifecycle(path, "new", enabled=True)
    new.start({"run": {}}, resume_expected=False)
    assert _read(path)["run"]["segments"] == []


@pytest.mark.parametrize(
    "env,kind",
    [
        ({"MOLCRAWL_LAUNCHER": "srun_deepspeed", "RANK": "0"}, "srun_deepspeed"),
        ({"TORCHELASTIC_RUN_ID": "x", "RANK": "0", "LOCAL_WORLD_SIZE": "4"}, "torchrun_legacy"),
        ({}, "single_process"),
        ({"RANK": "0"}, "other"),
    ],
)
def test_launcher_type(env, kind):
    rec = p.launcher_record(environ=env, query_scheduler=False)
    assert rec["type"] == kind
    assert rec["type_from"]


def test_launcher_script_hash(tmp_path):
    script = tmp_path / "launch.sbatch"
    script.write_text("#!/bin/bash\n")
    rec = p.launcher_record(
        environ={"MOLCRAWL_LAUNCHER": "srun_deepspeed", "MOLCRAWL_LAUNCHER_SCRIPT": str(script)},
        query_scheduler=False,
    )
    assert rec["script_path"] == str(script)
    assert rec["script_sha256"] == p.sha256_file(str(script))


# ------------------------------------------------------- GPT-2 schema 2 ---- #

def _gpt2_write(tmp_path, **extra):
    return gm.write_manifest(
        str(tmp_path), {"learning_rate": 1e-4}, {"learning_rate": 6e-4},
        data={"modality": "x"},
        batch={"batch_size": 8, "gradient_accumulation_steps_configured": 320,
               "gradient_accumulation_steps_per_rank": 80, "world_size": 4, "block_size": 1024},
        schedule={}, objective={}, evaluation={}, selection={}, seed={},
        collect_runtime=False, **extra,
    )


def test_gpt2_manifest_keeps_schema1_fields_and_adds_schema2(tmp_path):
    m = _gpt2_write(tmp_path)
    for key in ("written", "framework", "run", "placement", "data", "batch", "schedule", "objective",
                "eval", "selection", "seed", "sources", "introduced", "env"):
        assert key in m, key
    assert m["schema_version"] == p.SCHEMA_VERSION
    assert m["deepspeed"] == {"enabled": False}
    for key in ("env_by_prefix", "launcher", "provenance", "batch_policy", "model", "optimizer",
                "slurm", "cpu", "runtime", "gpu"):
        assert key in m, key
    assert m["batch"]["effective_global_batch"] == 2560


def test_note_resume_through_a_lifecycle_survives_completion(tmp_path):
    life = p.RunLifecycle(str(tmp_path / gm.MANIFEST), "r", enabled=True)
    life.start({"run": {}})
    _gpt2_write(tmp_path, lifecycle=life)
    gm.note_resume(str(tmp_path), 500, lifecycle=life, checkpoint="/c/checkpoint-500", world_size_after=4)
    life.complete(reached_step=600)
    doc = _read(str(tmp_path / gm.MANIFEST))
    assert doc["run"]["resume_history"][-1]["from_iter"] == 500
    assert doc["run"]["resume_history"][-1]["checkpoint"] == "/c/checkpoint-500"
    assert doc["run"]["status"] == "completed"


def test_build_provenance_separates_config_from_cli(tmp_path):
    cfg = tmp_path / "cfg.py"
    cfg.write_text("learning_rate = 1e-4\n")
    trainer = tmp_path / "trainer.py"
    trainer.write_text("print(learning_rate, seed)\n")
    prov = gm.build_provenance(
        argv=[str(cfg), "--seed=7"],
        trainer_file=str(trainer),
        configurator_path=None,
        defaults={"learning_rate": 6e-4, "seed": 1337},
        after_config_file={"learning_rate": 1e-4, "seed": 1337, "introduced_only": 3},
        after_cli={"learning_rate": 1e-4, "seed": 7, "introduced_only": 3},
        resolved={"learning_rate": 1e-4, "seed": 7, "introduced_only": 3},
        read_sources=[str(trainer)],
    )
    assert prov["config_file"]["sha256"] == p.sha256_file(str(cfg))
    assert prov["keys"]["learning_rate"]["set_by"] == "after_config_file"
    assert prov["keys"]["seed"]["set_by"] == "after_cli"
    assert prov["used"]["introduced_only"]["used"] is False
    assert prov["used"]["seed"]["used"] is True
    assert "after_deepspeed" not in prov["stages_recorded"]
