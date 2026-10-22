"""A continuation after an external cancellation, and only when it is asked for by id.

The continuation guard accepts TIMEOUT alone, because afterany fires on a crash and on
a time-out alike and the guard cannot tell them apart. A cancellation is the one end
state whose cause a resume will not repeat -- when it came from outside the run, such as
the maintenance window of 2026-10-21 that stopped jobs 187299 and 192549 mid-schedule --
and the guard still cannot tell that from a cancellation that was a judgement about the
run. So it is said explicitly, by job id.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "workflows" / "bert-grid.sbatch"

CONFIG = '''\
max_steps = 15000
warmup_steps = 1500
learning_rate = 0.001
seed = 42
batch_size = 32
gradient_accumulation_steps = 20
per_device_eval_batch_size = 32
expected_global_batch = 2560
log_interval = 100
save_steps = 1000
max_length = 1024
document_masking = True
save_on_improve = True
bf16 = False
dataloader_num_workers = 0
dataloader_pin_memory = False
adam_beta2 = 0.999
dataset_dir = "{dataset_dir}"
model_path = "{model_path}"
'''


@pytest.fixture
def world(tmp_path):
    py = os.environ.get("PYTHON") or sys.executable
    if not (py and os.access(py, os.X_OK)):
        pytest.skip("no interpreter to run the launcher's config dump with")

    src = tmp_path / "learning_source"
    (src / "compounds").mkdir(parents=True)
    out_root = tmp_path / "runs"
    out = out_root / "compounds" / "bert-output" / "probe-lr1e3"
    ckpt = out / "checkpoint-9700"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"max_steps": 15000, "global_step": 9700}), encoding="utf-8")
    (out / "segments.log").write_text("1 187299 2026-10-20T00:00:00+09:00\n", encoding="utf-8")

    cfg_dir = tmp_path / "configs" / "compounds"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "probe_lr1e3.py"
    cfg.write_text(CONFIG.format(dataset_dir=str(src / "compounds"), model_path=str(out)),
                   encoding="utf-8")

    bindir = tmp_path / "bin"
    bindir.mkdir()

    def run(prev_state, resume_after=None, prev_job="187299"):
        # Stand in for Slurm's accounting, which is what the guard reads.
        (bindir / "sacct").write_text(f"#!/bin/sh\necho {prev_state}\n", encoding="utf-8")
        (bindir / "sacct").chmod(0o755)
        env = dict(os.environ)
        env.update(
            CONFIG=str(cfg), SEGMENT="2", MODALITY="compounds",
            LEARNING_SOURCE_DIR=str(src), MODEL_OUTPUT_ROOT=str(out_root),
            PYTHON=py, TORCHRUN=shutil.which("true") or "/bin/true",
            SLURM_SUBMIT_DIR=str(ROOT), PYTHONPATH=str(ROOT),
            PREV_JOB=prev_job,
            PATH=f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        env.pop("RESUME_AFTER_CANCELLED", None)
        if resume_after is not None:
            env["RESUME_AFTER_CANCELLED"] = resume_after
        return subprocess.run(["bash", str(SCRIPT)], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, timeout=600)

    return run


def test_cancelled_alone_is_still_refused(world):
    r = world("CANCELLED")
    assert r.returncode != 0
    assert "ended CANCELLED, not TIMEOUT" in r.stderr
    assert "RESUME_AFTER_CANCELLED=187299" in r.stderr, "the message must say how"


def test_naming_the_job_lets_it_through(world):
    r = world("CANCELLED", resume_after="187299")
    combined = r.stdout + r.stderr
    assert "was CANCELLED" in combined, combined[-1500:]
    assert "RESUME_AFTER_CANCELLED names it" in combined, combined[-1500:]
    assert "REFUSING" not in r.stderr


def test_naming_a_different_job_does_not(world):
    """A value left in the environment must not wave the next segment through."""
    r = world("CANCELLED", resume_after="999999")
    assert r.returncode != 0
    assert "ended CANCELLED, not TIMEOUT" in r.stderr


def test_failed_stays_refused_even_when_named(world):
    """FAILED is a state whose cause a resume can repeat; this is for CANCELLED only."""
    r = world("FAILED", resume_after="187299")
    assert r.returncode != 0
    assert "ended FAILED, not TIMEOUT" in r.stderr
    assert "may have stopped for a reason a resume would repeat" in r.stderr


def test_timeout_still_needs_nothing(world):
    r = world("TIMEOUT")
    combined = r.stdout + r.stderr
    assert "ended TIMEOUT" in combined
    assert "REFUSING" not in r.stderr
