"""gpt2/train.py on CPU, a few steps: the manifest a real run leaves behind.

Three runs of the actual trainer in a subprocess, with a 1-layer model on a
16-token toy dataset:

completed
    starting -> running -> completed, the schema-1 fields still present, the
    config-file stage separated from the CLI stage, the batch-policy record
    agreeing with train.py's own division.
failed
    a dataset directory that does not exist: the run fails after the first
    manifest write, and the manifest says failed, in which phase, and why.
resumed
    the completed run continued to more steps: the previous segment is kept and
    resume_history names the checkpoint.

CPU only and seconds long, but it starts a Python process that imports torch
three times, so it is marked integration rather than living in tests/unit.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(REPO, "molcrawl", "models", "gpt2", "train.py")


def _dataset(root):
    from datasets import Dataset

    rows = [[(i + j) % 31 + 1 for j in range(17)] for i in range(64)]
    for split in ("train", "valid"):
        Dataset.from_dict({"input_ids": rows}).save_to_disk(os.path.join(root, f"{split}.arrow"))
    return root


def _config(path, out_dir, dataset_dir, **overrides):
    values = {
        "out_dir": out_dir,
        "dataset": "toy",
        "dataset_params": {"dataset_dir": dataset_dir},
        "meta_vocab_size": 32,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "block_size": 16,
        "dropout": 0.0,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_iters": 4,
        "lr_decay_iters": 4,
        "warmup_iters": 1,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "eval_interval": 2,
        "eval_iters": 1,
        "log_interval": 1,
        "always_save_checkpoint": True,
        "init_from": "scratch",
        "device": "cpu",
        "dtype": "float32",
        "compile": False,
    }
    values.update(overrides)
    with open(path, "w") as fh:
        for key, value in values.items():
            fh.write(f"{key} = {value!r}\n")
    return path


def _run(config_path, *cli, cwd):
    env = {k: v for k, v in os.environ.items() if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")}
    env.update(
        PYTHONPATH=REPO + os.pathsep + env.get("PYTHONPATH", ""),
        CUDA_VISIBLE_DEVICES="",
        USE_WANDB="false",
        WANDB_MODE="disabled",
        HF_DATASETS_OFFLINE="1",
        SUBSET_TEST_MARKER="present",
        HF_TOKEN="should-not-appear",
    )
    return subprocess.run(
        [sys.executable, TRAIN, config_path, *cli],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=600,
    )


def _manifest(out_dir):
    with open(os.path.join(out_dir, "run_manifest.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("gpt2cpu")
    data = _dataset(str(root / "data"))
    out = str(root / "out")
    cfg = _config(str(root / "cfg.py"), out, data)
    done = _run(cfg, "--learning_rate=0.002", cwd=str(root))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return {"root": root, "data": data, "out": out, "cfg": cfg, "manifest": _manifest(out)}


def test_completed_run_records_its_lifecycle(completed_run):
    m = completed_run["manifest"]
    assert m["schema_version"] == 2
    assert m["run"]["status"] == "completed"
    assert [h["status"] for h in m["run"]["status_history"]] == ["starting", "running", "completed"]
    assert m["run"]["exit_code"] == 0
    assert m["run"]["reached_step"] == 5  # iterations 0..4 inclusive, max_iters=4
    assert m["run"]["stop_reason"] == "max_iters"
    assert m["run"]["final_checkpoint"].endswith("checkpoint-4")
    assert m["run"]["best_metric"]["name"] == "val_loss"


def test_schema1_fields_are_still_there(completed_run):
    m = completed_run["manifest"]
    for key in ("sources", "introduced", "env", "placement", "data", "batch", "schedule",
                "objective", "eval", "selection", "seed"):
        assert key in m, key
    assert m["batch"]["effective_global_batch"] == 2
    assert m["data"]["rows"] == {"train": 64, "eval": 64}


def test_config_file_and_cli_are_separate_stages(completed_run):
    prov = completed_run["manifest"]["provenance"]
    assert prov["config_file"]["path"] == completed_run["cfg"]
    assert len(prov["config_file"]["sha256"]) == 64
    assert prov["keys"]["learning_rate"]["set_by"] == "after_cli"
    assert prov["keys"]["learning_rate"]["stages"]["after_config_file"] == 1e-3
    assert prov["keys"]["n_embd"]["set_by"] == "after_config_file"
    assert prov["keys"]["seed"]["set_by"] == "defaults"
    assert prov["cli_overrides"] == [{"key": "learning_rate", "value": "0.002", "position": 1}]
    # meta_vocab_size is read by train.py; a config-only name would be used: false.
    assert prov["used"]["meta_vocab_size"]["used"] is True


def test_batch_policy_agrees_with_train_py(completed_run):
    bp = completed_run["manifest"]["batch_policy"]
    assert bp["batch_policy"] == "global_fixed"
    assert bp["matches_legacy"] is True
    assert bp["effective_global_batch"] == 2
    # Target 2 at micro batch 2 cannot be spread over even one 4-GPU node
    # (2 / (2 x 4) = 0.25), and the record says so rather than rounding.
    one_node = bp["scaling_feasibility"]["candidates"][0]
    assert (one_node["nodes"], one_node["feasible"], one_node["quotient"]) == (1, False, 0.25)
    assert "< 1" in one_node["reason"]
    assert bp["scaling_feasibility"]["max_feasible_candidate_node_count"] is None


def test_derived_values_match_the_toy_run(completed_run):
    d = completed_run["manifest"]["derived"]
    assert d["sequences_per_optimizer_step"] == 2
    assert d["optimizer_steps_planned"] == 5
    assert d["train_rows"] == 64
    assert d["epochs_planned"] == pytest.approx(5 * 2 / 64)
    lr = d["learning_rate"]
    assert lr["initial"] == 0.0
    assert lr["peak"] == pytest.approx(2e-3)  # the CLI override, not the config file
    assert lr["floor_reached_at_step"] == 4 and lr["floor_reached_within_run"] is True
    assert lr["at_final_step"] == pytest.approx(1e-4)


def test_model_optimizer_and_environment_sections(completed_run):
    m = completed_run["manifest"]
    assert m["model"]["class"] == "GPT"
    assert m["model"]["parameter_count"] > 0
    assert m["model"]["attention_implementation"] in ("sdpa (is_causal)", "manual causal matmul")
    assert m["optimizer"]["class"] == "AdamW"
    assert m["optimizer"]["fused"] in (None, False)  # CPU: configure_optimizers does not ask for fused
    assert m["deepspeed"] == {"enabled": False}
    assert m["launcher"]["type"] == "single_process"
    assert m["runtime"]["torch"]
    assert m["env_by_prefix"]["config"]["SUBSET_TEST_MARKER"] == "present"
    assert m["env_by_prefix"]["runtime"]["HF_TOKEN"] == "<redacted>"
    assert "should-not-appear" not in json.dumps(m)


def test_failed_run_is_recorded_as_failed(tmp_path):
    out = str(tmp_path / "out")
    cfg = _config(str(tmp_path / "cfg.py"), out, str(tmp_path / "no-such-dataset"))
    done = _run(cfg, cwd=str(tmp_path))
    assert done.returncode != 0
    m = _manifest(out)
    assert m["run"]["status"] == "failed"
    assert m["run"]["failure"]["phase"] == "setup"
    assert m["run"]["failure"]["error_summary"]
    assert [h["status"] for h in m["run"]["status_history"]] == ["starting", "failed"]


def test_resumed_run_keeps_the_previous_segment(completed_run):
    root = completed_run["root"]
    cfg = _config(str(root / "cfg_resume.py"), completed_run["out"], completed_run["data"],
                  init_from="resume", max_iters=6, lr_decay_iters=6)
    done = _run(cfg, cwd=str(root))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    m = _manifest(completed_run["out"])
    assert m["run"]["status"] == "completed"
    assert m["run"]["segments"][-1]["run_id"] == completed_run["manifest"]["run"]["run_id"]
    assert m["run"]["segments"][-1]["status"] == "completed"
    entry = m["run"]["resume_history"][-1]
    assert entry["checkpoint"].endswith(os.path.join("checkpoint-4", "training_state.bin"))
    assert entry["optimizer_state_restored"] is True
    assert entry["world_size_after"] == 1
    assert m["model"]["initialization"] == "resume"


def test_deepspeed_config_on_cpu_single_process_fails_before_training(tmp_path):
    """The backend refuses a launch it cannot run faithfully, and the manifest says why."""
    data = _dataset(str(tmp_path / "data"))
    out = str(tmp_path / "out")
    cfg = _config(str(tmp_path / "cfg.py"), out, data)
    ds = tmp_path / "cfg.deepspeed.json"
    ds.write_text("{}")
    done = _run(cfg, f"--deepspeed_config={ds}", cwd=str(tmp_path))
    assert done.returncode != 0
    m = _manifest(out)
    assert m["run"]["status"] == "failed"
    assert m["deepspeed"]["enabled"] is True
    assert m["deepspeed"]["config_path"] == str(ds)
    assert "distributed launch" in m["run"]["failure"]["error_summary"]


def test_data_splits_and_tokenizer_are_described(completed_run):
    data = completed_run["manifest"]["data"]
    train = data["splits"]["train"]
    assert train["loading_method"] == "load_from_disk(<dir>/<split>.arrow)"
    assert train["path"].endswith("train.arrow")
    assert train["rows"] == 64 and train["row_length"] == 17
    assert train["fingerprint"]
    assert data["splits"]["eval"]["path"].endswith("valid.arrow")
    assert data["tokenizer"]["class"] is None and data["tokenizer"]["vocab_size_used_by_model"] == 32
    assert data["preparation"]["commit"] is None and data["preparation"]["reason"]
