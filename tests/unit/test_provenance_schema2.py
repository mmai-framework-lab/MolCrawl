"""Schema-2 helpers in ``_provenance``: argv order, stages, used keys, env, writes."""

import json
import os
import textwrap

import pytest

from molcrawl.models import _provenance as p


# ---------------------------------------------------------------- argv ------ #

def test_config_file_then_overrides_is_accepted(tmp_path):
    cfg = tmp_path / "cfg.py"
    cfg.write_text("learning_rate = 1e-4\n")
    parsed = p.validate_argv([str(cfg), "--learning_rate=3e-4", "--seed=7"])
    assert parsed["problems"] == []
    assert [o["key"] for o in parsed["cli_overrides"]] == ["learning_rate", "seed"]
    record = p.config_file_record(parsed)
    assert record["path"] == str(cfg)
    assert record["sha256"] == p.sha256_file(str(cfg))
    assert len(record["sha256"]) == 64


def test_config_file_after_an_override_is_refused(tmp_path):
    cfg = tmp_path / "cfg.py"
    cfg.write_text("")
    with pytest.raises(p.ArgvOrderError, match="comes after --key=value"):
        p.validate_argv(["--seed=7", str(cfg)])


def test_two_config_files_are_refused(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("")
    b.write_text("")
    with pytest.raises(p.ArgvOrderError, match="at most one"):
        p.validate_argv([str(a), str(b)])


def test_no_config_file_is_recorded_as_such():
    record = p.config_file_record(p.parse_argv(["--seed=1"]))
    assert record["path"] is None and record["reason"]


def test_missing_config_file_has_no_hash_rather_than_raising(tmp_path):
    parsed = p.parse_argv([str(tmp_path / "absent.py")])
    assert parsed["config_files"][0]["sha256"] is None


def test_relative_config_uses_the_configurator_fallback(tmp_path, monkeypatch):
    # configurator.py tries <dir of trainer's parent>/<arg> when <arg> does not exist from cwd.
    trainer_dir = tmp_path / "models" / "gpt2"
    trainer_dir.mkdir(parents=True)
    trainer = trainer_dir / "train.py"
    trainer.write_text("")
    target = tmp_path / "models" / "cfgs"
    target.mkdir()
    (target / "x.py").write_text("")
    monkeypatch.chdir(tmp_path)
    assert p.resolve_config_path("cfgs/x.py", str(trainer)) == str(target / "x.py")


# -------------------------------------------------------------- stages ------ #

def test_stage_sources_attributes_each_value_to_the_stage_that_set_it():
    snaps = {
        "defaults": {"lr": 6e-4, "seed": 42, "bf16": False},
        "after_config_file": {"lr": 1e-4, "seed": 42, "bf16": False, "extra": 3},
        "after_cli": {"lr": 3e-4, "seed": 42, "bf16": False, "extra": 3},
        "after_deepspeed": None,
        "resolved": {"lr": 3e-4, "seed": 42, "bf16": False, "extra": 3},
    }
    out = p.stage_sources(snaps)
    assert out["lr"]["set_by"] == "after_cli"
    assert out["lr"]["final"] == 3e-4
    assert [h["stage"] for h in out["lr"]["overridden"]] == ["after_config_file", "after_cli"]
    assert out["seed"]["set_by"] == "defaults" and out["seed"]["overridden"] == []
    assert out["extra"]["set_by"] == "after_config_file"
    assert "after_deepspeed" not in out["lr"]["stages"]


def test_scalar_snapshot_keeps_the_config_keys_filter():
    ns = {"a": 1, "_b": 2, "c": [1], "d": None, "e": "x", "f": 1.5, "g": True}
    assert p.scalar_snapshot(ns, exclude={"e"}) == {"a": 1, "f": 1.5, "g": True}


# ---------------------------------------------------------------- used ------ #

def test_names_read_by_finds_plain_loads_and_globals_get(tmp_path):
    src = tmp_path / "trainer.py"
    src.write_text(textwrap.dedent(
        """
        learning_rate = 1e-4
        x = learning_rate * 2
        y = globals().get("bf16", False)
        z = globals()["tf32"]
        """
    ))
    names = p.names_read_by([str(src)])
    assert {"learning_rate", "bf16", "tf32"} <= names


def test_mark_used_flags_names_no_code_reads():
    marked = p.mark_used({"learning_rate": 1e-4, "_GLOBAL_BATCH_typo": 1, "bf16": True}, {"learning_rate", "bf16"})
    assert marked["learning_rate"]["used"] is True
    assert marked["_GLOBAL_BATCH_typo"] == {"value": 1, "used": False}


# ----------------------------------------------------------------- env ------ #

def test_environment_by_prefix_catches_new_names_and_withholds_secrets():
    env = {
        "SUBSET_BERT_EPOCHS": "9",
        "SUBSET_SOMETHING_NEW": "x",
        "WANDB_API_KEY": "hunter2",
        "HF_TOKEN": "hf_abc",
        "NCCL_DEBUG": "INFO",
        "HOME": "/somewhere",
    }
    out = p.environment_by_prefix(environ=env)
    assert out["config"]["SUBSET_BERT_EPOCHS"] == "9"
    assert out["config"]["SUBSET_SOMETHING_NEW"] == "x"
    assert out["config"]["WANDB_API_KEY"] == p.REDACTED
    assert out["runtime"]["HF_TOKEN"] == p.REDACTED
    assert out["runtime"]["NCCL_DEBUG"] == "INFO"
    assert "HOME" not in out["config"] and "HOME" not in out["runtime"]
    assert out["redacted"] == ["HF_TOKEN", "WANDB_API_KEY"]
    assert "hunter2" not in json.dumps(out) and "hf_abc" not in json.dumps(out)


# --------------------------------------------------------------- write ------ #

def test_write_json_atomic_replaces_and_leaves_no_temp(tmp_path):
    path = tmp_path / "run_manifest.json"
    p.write_json_atomic(str(path), {"a": 1})
    p.write_json_atomic(str(path), {"a": 2, "s": {1, 2}, "t": (3,)})
    data = json.loads(path.read_text())
    assert data["a"] == 2 and sorted(data["s"]) == [1, 2] and data["t"] == [3]
    assert [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")] == []


def test_update_json_atomic_mutates_in_place(tmp_path):
    path = tmp_path / "m.json"
    assert p.update_json_atomic(str(path), lambda d: None) is None
    p.write_json_atomic(str(path), {"run": {"status": "starting"}})
    p.update_json_atomic(str(path), lambda d: d["run"].update(status="running"))
    assert json.loads(path.read_text())["run"]["status"] == "running"


@pytest.mark.parametrize("rank,expected", [(None, True), ("0", True), ("3", False)])
def test_is_rank_zero_reads_rank_before_the_process_group_exists(monkeypatch, rank, expected):
    if rank is None:
        monkeypatch.delenv("RANK", raising=False)
    else:
        monkeypatch.setenv("RANK", rank)
    assert p.is_rank_zero() is expected


# ------------------------------------------------------------- runtime ------ #

def test_runtime_versions_has_every_field_even_when_absent():
    out = p.runtime_versions()
    for key in ("python", "torch", "torch_cuda", "nccl", "transformers", "accelerate",
                "datasets", "pyarrow", "numpy", "deepspeed"):
        assert key in out


def test_cpu_and_slurm_info_do_not_raise_outside_slurm(monkeypatch):
    for name in list(os.environ):
        if name.startswith("SLURM_"):
            monkeypatch.delenv(name, raising=False)
    cpu = p.cpu_info()
    assert cpu["cpus_per_task_requested"] is None
    assert cpu["cpus_usable_by_process"] is None or cpu["cpus_usable_by_process"] >= 1
    assert p.slurm_info()["job_id"] is None


def test_gpu_info_records_reason_instead_of_raising():
    out = p.gpu_info(query_driver=False)
    assert "devices" in out and "error" in out
