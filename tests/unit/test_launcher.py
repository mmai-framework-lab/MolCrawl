"""The srun launcher (DeepSpeed order §8) and its sbatch / submitter, without Slurm."""

import argparse
import json
import os
import re

import pytest

from molcrawl.models import _launcher as L

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SBATCH = os.path.join(REPO, "workflows", "deepspeed-train.sbatch")


@pytest.mark.parametrize("nodes", [1, 2, 4, 8])
def test_sbatch_args_one_task_per_gpu(nodes):
    args = L.sbatch_args(nodes=nodes, cpus_per_task=6, time="02:00:00", job_name="x", account="acct")
    assert f"--nodes={nodes}" in args
    assert f"--ntasks={4 * nodes}" in args
    assert "--ntasks-per-node=4" in args
    assert f"--gpus={4 * nodes}" in args
    assert "--gpu-bind=none" in args
    assert "--cpus-per-task=6" in args
    assert "--account=acct" in args
    assert not any(a.startswith(("--gpus-per-node", "--gres")) for a in args)


def test_account_is_not_written_unless_given():
    assert not any(a.startswith("--account") for a in L.sbatch_args(nodes=1, cpus_per_task=2, time="10", job_name="x"))


@pytest.mark.parametrize("bad", [dict(nodes=0), dict(cpus_per_task=0), dict(time="two hours")])
def test_sbatch_args_refuse_bad_values(bad):
    kw = dict(nodes=1, cpus_per_task=2, time="01:00:00", job_name="x")
    kw.update(bad)
    with pytest.raises(L.LaunchError):
        L.sbatch_args(**kw)


def test_srun_args_restate_cpus_per_task():
    args = L.srun_args(nodes=2, cpus_per_task=6)
    assert args[0] == "srun"
    assert {"--nodes=2", "--ntasks=8", "--ntasks-per-node=4", "--cpus-per-task=6", "--gpu-bind=none"} <= set(args)


@pytest.mark.parametrize("workers,cpus", [(0, 2), (2, 4), (4, 6)])
def test_cpus_per_task_follow_dataloader_workers(workers, cpus):
    assert L.cpus_per_task_for(workers) == cpus


@pytest.mark.parametrize(
    "nodelist,hosts",
    [("c187", ["c187"]),
     ("c[001-003]", ["c001", "c002", "c003"]),
     ("c[001-002,010],d7", ["c001", "c002", "c010", "d7"]),
     ("c[98-101]", ["c98", "c99", "c100", "c101"])],
)
def test_expand_nodelist(nodelist, hosts):
    assert L.expand_nodelist(nodelist) == hosts


def _slurm(**over):
    env = {"SLURM_PROCID": "5", "SLURM_LOCALID": "1", "SLURM_NTASKS": "8", "SLURM_JOB_NUM_NODES": "2",
           "SLURM_NTASKS_PER_NODE": "4(x2)", "SLURM_JOB_NODELIST": "c[010-011]", "SLURM_JOB_ID": "123456"}
    env.update(over)
    return env


def test_distributed_env_maps_slurm_to_torch():
    env = L.distributed_env(_slurm(), resolve_ip=lambda h: "10.0.0.10", host_of=lambda nl: L.expand_nodelist(nl)[0])
    assert (env["RANK"], env["LOCAL_RANK"], env["WORLD_SIZE"]) == ("5", "1", "8")
    assert env["MASTER_ADDR"] == "10.0.0.10"
    assert env["MOLCRAWL_RENDEZVOUS_HOST"] == "c010"
    assert env["MASTER_PORT"] == str(20000 + 123456 % 20000)
    assert env["MOLCRAWL_LAUNCHER"] == "srun_deepspeed"


@pytest.mark.parametrize(
    "override,fragment",
    [({"SLURM_NTASKS": "2"}, "one task per GPU"),
     ({"SLURM_NTASKS_PER_NODE": "1(x2)"}, "SLURM_NTASKS_PER_NODE"),
     ({"SLURM_LOCALID": "4"}, "SLURM_LOCALID"),
     ({"SLURM_PROCID": "8"}, "SLURM_PROCID")],
)
def test_distributed_env_refuses_other_layouts(override, fragment):
    with pytest.raises(L.LaunchError, match=fragment):
        L.distributed_env(_slurm(**override), resolve_ip=lambda h: "10.0.0.1", host_of=lambda nl: "c010")


def test_distributed_env_outside_srun_is_refused():
    with pytest.raises(L.LaunchError, match="inside srun"):
        L.distributed_env({}, resolve_ip=lambda h: "x", host_of=lambda nl: "x")


def test_master_port_is_stable_per_job_and_in_range():
    ports = {L.master_port_for(str(j)) for j in (1, 20001, 122187)}
    assert all(20000 <= p < 40000 for p in ports)
    assert L.master_port_for("122187") == L.master_port_for("122187")


def test_main_prints_srun_args(capsys):
    assert L.main(["srun-args", "--nodes", "1", "--cpus-per-task", "4"]) == 0
    lines = capsys.readouterr().out.split()
    assert lines[0] == "srun" and "--cpus-per-task=4" in lines


def test_record_failure_writes_a_preliminary_manifest(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    L.record_launch_failure(str(out), 1, environ={"SLURM_JOB_ID": "9", "MOLCRAWL_LAUNCHER": "srun_deepspeed"})
    doc = json.loads((out / "run_manifest.json").read_text())
    assert doc["preliminary"] is True
    assert doc["run"]["status"] == "failed"
    assert doc["run"]["failure"]["phase"] == "launch"
    assert doc["run"]["exit_code"] == 1
    assert doc["launcher"]["type"] == "srun_deepspeed"


@pytest.mark.parametrize("status,expected", [("running", "failed"), ("starting", "failed"), ("completed", "completed")])
def test_record_failure_only_overrides_a_non_terminal_status(tmp_path, status, expected):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"run": {"status": status, "status_history": [{"status": status}]}}))
    L.record_launch_failure(str(tmp_path), 137, environ={})
    doc = json.loads(path.read_text())
    assert doc["run"]["status"] == expected
    if expected == "failed":
        assert doc["run"]["failure"]["last_status"] == status
        assert doc["run"]["exit_code"] == 137


# ------------------------------------------------------------ sbatch file ---- #

def test_sbatch_file_follows_the_placement_and_path_rules():
    text = open(SBATCH).read()
    directives = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert not any(re.search(r"--(gres|gpus-per-node|account|nodes|gpus=)", d) for d in directives)
    assert "molcrawl.models._launcher srun-args" in text
    assert "--cpus-per-task" in text
    assert ': "${RUNS_ROOT:?' in text and ': "${LEARNING_SOURCE_DIR:?' in text
    assert "git-common-dir" in text
    assert "record-failure" in text
    # No absolute server path, account or user name in tracked launch code.
    assert not re.search(r"""(^|[\s'"=])/(data\d*|home|work)/""", text, re.M)
    assert "--account" not in text
    # The header may mention the legacy path; no command line may use it.
    commands = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert not any("torchrun" in line for line in commands)


# ------------------------------------------------------------- submitter ---- #

def _submit_args(**over):
    kw = dict(arch="gpt2", model_config="molcrawl/tasks/pretrain/configs/rna/gpt2_small.py", run_name="t",
              nodes=1, time="01:00:00", account=None, partition="gpu", cpus_per_task=None, ds_config=None,
              job_name=None, extra_args=None, expect_commit=None, submit=False)
    kw.update(over)
    return argparse.Namespace(**kw)


@pytest.fixture
def submit_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("submit_deepspeed", os.path.join(REPO, "scripts", "submit_deepspeed.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ds_file(tmp_path, body=None):
    from molcrawl.models._deepspeed_config import default_config

    path = tmp_path / "x.deepspeed.json"
    path.write_text(json.dumps(body if body is not None else default_config()))
    return str(path)


def test_submitter_plan_for_one_and_eight_nodes(submit_module, tmp_path):
    env = {"RUNS_ROOT": str(tmp_path / "runs"), "LEARNING_SOURCE_DIR": str(tmp_path)}
    one = submit_module.plan(_submit_args(ds_config=_ds_file(tmp_path)), environ=env)
    assert one["problems"] == []
    assert one["batch"]["gradient_accumulation_steps_per_rank"] == 40  # rna gpt2_small: 16 x 160
    assert "--ntasks=4" in one["command"] and one["command"][-1] == "workflows/deepspeed-train.sbatch"
    eight = submit_module.plan(_submit_args(ds_config=_ds_file(tmp_path), nodes=8), environ=env)
    assert eight["problems"] == []
    assert eight["batch"]["gradient_accumulation_steps_per_rank"] == 5
    sixteen = submit_module.plan(_submit_args(ds_config=_ds_file(tmp_path), nodes=16), environ=env)
    assert any("batch" in p for p in sixteen["problems"])


def test_submitter_reports_missing_roots_and_bad_ds_config(submit_module, tmp_path):
    bad = _ds_file(tmp_path, {"optimizer": {"type": "AdamW"}})
    result = submit_module.plan(_submit_args(ds_config=bad), environ={})
    text = " ".join(result["problems"])
    assert "RUNS_ROOT is not set" in text and "LEARNING_SOURCE_DIR is not set" in text
    assert "optimizer" in text


def test_submitter_refuses_undeclared_bert_batch(submit_module, tmp_path):
    env = {"RUNS_ROOT": str(tmp_path), "LEARNING_SOURCE_DIR": str(tmp_path)}
    result = submit_module.plan(_submit_args(arch="bert", ds_config=_ds_file(tmp_path),
                                model_config="molcrawl/tasks/pretrain/configs/compounds/bert_small.py"), environ=env)
    assert any("expected_global_batch" in p for p in result["problems"])


def test_submitter_refuses_commas_in_extra_args(submit_module, tmp_path):
    env = {"RUNS_ROOT": str(tmp_path), "LEARNING_SOURCE_DIR": str(tmp_path)}
    result = submit_module.plan(_submit_args(ds_config=_ds_file(tmp_path), extra_args="--a=1,2"), environ=env)
    assert any("comma" in p for p in result["problems"])


def test_sbatch_guard_matches_the_trainers_that_declare_deepspeed_config():
    """The sbatch refuses a trainer without the key; gpt2 has it, bert does not yet."""
    pattern = re.compile(r"^\s*deepspeed_config\s*(:[^=]*)?=", re.M)
    gpt2 = open(os.path.join(REPO, "molcrawl", "models", "gpt2", "train.py")).read()
    bert = open(os.path.join(REPO, "molcrawl", "models", "bert", "main.py")).read()
    assert pattern.search(gpt2)
    assert not pattern.search(bert)
