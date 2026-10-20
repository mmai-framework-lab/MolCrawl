"""EXTEND=1 opens two guards, and only on the terms that make an extension safe.

The launcher refuses a continuation whose max_steps differs from the checkpoint's, and
refuses one whose predecessor did not end TIMEOUT. An extension -- continuing a
finished run to a larger max_steps to see whether the loss flattens -- needs both, so
it is gated on a constant learning rate: HF decays over max_steps with no separate
decay length, so a decaying schedule would be recomputed and the continuation would be
a different run.

These drive the real script. Each case stops at a guard, so no GPU is involved.
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

CONFIG_TEMPLATE = '''\
max_steps = {max_steps}
warmup_steps = 100
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
{scheduler}
dataset_dir = "{dataset_dir}"
model_path = "{model_path}"
'''


def _python():
    """The interpreter running these tests, which is the one the launcher would use."""
    p = os.environ.get("PYTHON") or sys.executable
    return p if p and os.access(p, os.X_OK) else None


@pytest.fixture
def world(tmp_path):
    """A config under configs/<modality>/, an input tree, and one finished checkpoint."""
    py = _python()
    if py is None:
        pytest.skip("no interpreter to run the launcher's config dump with")

    src = tmp_path / "learning_source"
    (src / "compounds").mkdir(parents=True)
    out_root = tmp_path / "runs"
    out = out_root / "compounds" / "bert-output" / "probe-lr1e3"
    out.mkdir(parents=True)
    (out / "segments.log").write_text("1 999999 2026-10-20T00:00:00+09:00\n", encoding="utf-8")

    def write_checkpoint(step, prev_max_steps, scheduler=False):
        ckpt = out / f"checkpoint-{step}"
        ckpt.mkdir(exist_ok=True)
        (ckpt / "trainer_state.json").write_text(
            json.dumps({"max_steps": prev_max_steps, "global_step": step}), encoding="utf-8"
        )
        sched = ckpt / "scheduler.pt"
        if scheduler:
            sched.write_bytes(b"not a real state dict")
        elif sched.exists():
            sched.unlink()

    cfg_dir = tmp_path / "configs" / "compounds"
    cfg_dir.mkdir(parents=True)

    # The continuation guard asks Slurm how the previous segment ended. Answer it here
    # so the cases below are decided by the gate and not by a real job id, and so a
    # missing one does not cost the guard's ten 30-second retries.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sacct = bindir / "sacct"
    sacct.write_text("#!/bin/sh\necho COMPLETED\n", encoding="utf-8")
    sacct.chmod(0o755)

    def write_config(max_steps, scheduler):
        line = f'lr_scheduler_type = "{scheduler}"' if scheduler else ""
        p = cfg_dir / "probe_lr1e3.py"
        p.write_text(
            CONFIG_TEMPLATE.format(
                max_steps=max_steps, scheduler=line,
                dataset_dir=str(src / "compounds"), model_path=str(out),
            ),
            encoding="utf-8",
        )
        return p

    def run(max_steps, scheduler=None, extend=None, segment="2",
            prev_done=15000, prev_max_steps=15000, keep_scheduler=False):
        write_checkpoint(prev_done, prev_max_steps, scheduler=keep_scheduler)
        cfg = write_config(max_steps, scheduler)
        env = dict(os.environ)
        env.update(
            CONFIG=str(cfg), SEGMENT=segment, MODALITY="compounds",
            LEARNING_SOURCE_DIR=str(src), MODEL_OUTPUT_ROOT=str(out_root),
            PYTHON=py, TORCHRUN=shutil.which("true") or "/bin/true",
            SLURM_SUBMIT_DIR=str(ROOT), PYTHONPATH=str(ROOT),
            PATH=f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        if extend is not None:
            env["EXTEND"] = extend
        else:
            env.pop("EXTEND", None)
        return subprocess.run(
            ["bash", str(SCRIPT)], env=env, cwd=str(ROOT),
            capture_output=True, text=True, timeout=600,
        )

    return run


def test_unset_still_refuses_a_different_max_steps(world):
    r = world(30000)
    assert r.returncode != 0
    assert "REFUSING" in r.stderr
    assert "max_steps=15000" in r.stderr and "max_steps=30000" in r.stderr


def test_extend_refuses_a_decaying_schedule(world):
    """Without a constant rate there is a decay length to recompute, so no extension."""
    r = world(30000, scheduler=None, extend="1")
    assert r.returncode != 0
    assert "lr_scheduler_type" in r.stderr
    assert "constant" in r.stderr


def test_extend_refuses_lowering_max_steps(world):
    """Downwards would still compress the schedule, constant rate or not."""
    r = world(10000, scheduler="constant", extend="1")
    assert r.returncode != 0
    assert "REFUSING" in r.stderr
    assert "raises max_steps, it does not lower it" in r.stderr


def test_extend_accepts_a_constant_schedule_raised_upwards(world):
    r = world(30000, scheduler="constant", extend="1")
    combined = r.stdout + r.stderr
    assert "extension: EXTEND=1, schedule constant" in combined, combined[-2000:]
    assert "max_steps 15000 -> 30000" in combined, combined[-2000:]
    assert "ended COMPLETED, extending it" in combined, combined[-2000:]


def test_a_completed_predecessor_is_still_refused_without_extend(world):
    """The relaxation is the only way COMPLETED is accepted.

    A checkpoint short of max_steps, so the schedule and the step count both pass and
    the previous segment's end state is what decides it.
    """
    r = world(15000, extend=None, prev_done=8000, prev_max_steps=15000)
    assert r.returncode != 0
    assert "ended COMPLETED, not TIMEOUT" in r.stderr
    assert "EXTEND=1 is how a finished run is extended" in r.stderr


def test_constant_with_warmup_is_also_accepted(world):
    r = world(30000, scheduler="constant_with_warmup", extend="1")
    combined = r.stdout + r.stderr
    assert "schedule constant_with_warmup" in combined, combined[-2000:]


def test_extend_refuses_an_inherited_scheduler_state(world):
    """base_lrs comes back with it, so the config's learning_rate would be ignored.

    LambdaLR.load_state_dict does self.__dict__.update(state_dict), which restores
    base_lrs and last_epoch. get_lr returns base_lr * lambda(last_epoch), so a run
    resuming with the old scheduler.pt continues at the original peak with the warmup
    already behind it. Measured on a real checkpoint-15000/scheduler.pt: base_lrs
    [0.001, 0.001], last_epoch 15000.
    """
    r = world(30000, scheduler="constant", extend="1", keep_scheduler=True)
    assert r.returncode != 0
    assert "scheduler.pt exists" in r.stderr
    assert "base_lrs" in r.stderr
    assert "keep optimizer.pt" in r.stderr
