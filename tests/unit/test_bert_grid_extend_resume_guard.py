"""The EXTEND=1 scheduler.pt guard must tell a new extension from a later segment.

The guard used to refuse on the file's presence alone, so every segment boundary of
an extension needed scheduler.pt removed by hand; four segments of the compounds
45,000-step extension came back FAILED for that reason. It now decides from what the
checkpoint was trained under.

The bash block is lifted out of workflows/bert-grid.sbatch and run as written, so
these cases exercise the shipped code rather than a restatement of it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

GUARD_START = '    if [ -f "${OUT}/checkpoint-${_last_pre}/scheduler.pt" ]; then'
GUARD_END = "    fi"


def repo_root() -> Path:
    """The checkout this test file belongs to.

    Not git's common dir: from a worktree that points at the main checkout, and the
    job script under test is the worktree's copy. workflows/unit-tests.sbatch uses the
    common dir on purpose, but for the interpreter, which a worktree does not hold.
    """
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def guard_block() -> str:
    """The guard exactly as it appears in the job script."""
    lines = (repo_root() / "workflows" / "bert-grid.sbatch").read_text(
        encoding="utf-8"
    ).splitlines()
    start = next(i for i, line in enumerate(lines) if line == GUARD_START)
    end = next(i for i, line in enumerate(lines[start + 1:], start + 1) if line == GUARD_END)
    return "\n".join(lines[start:end + 1])


def make_checkpoint(root: Path, step: int, max_steps: int, last_lr: float | None,
                    scheduler: bool = True, state: bool = True) -> Path:
    ck = root / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    if scheduler:
        (ck / "scheduler.pt").write_bytes(b"")
    if state:
        history = [] if last_lr is None else [{"step": step, "learning_rate": last_lr}]
        (ck / "trainer_state.json").write_text(
            json.dumps({"max_steps": max_steps, "global_step": step,
                        "log_history": history}), encoding="utf-8")
    return ck


def run_guard(guard_block: str, out: Path, step: int, cfg_steps: int,
              cfg_lr: float, tmp_path: Path,
              strict: bool = False) -> subprocess.CompletedProcess:
    """``strict`` runs it under the job script's own flags, ``set -eo pipefail``."""
    script = tmp_path / "guard.sh"
    script.write_text(
        ("set -eo pipefail\n" if strict else "set -u\n")
        + f'OUT={out}\n'
        f'_last_pre={step}\n'
        f'_cfg_steps={cfg_steps}\n'
        f'CFG=some-config.py\n'
        f'PYTHON={sys.executable}\n'
        f"_cfg_dump='max_steps={cfg_steps}\nlearning_rate={cfg_lr}'\n"
        + guard_block + "\n",
        encoding="utf-8",
    )
    return subprocess.run(["bash", str(script)], capture_output=True, text=True)


# (name, checkpoint max_steps, checkpoint last rate, config max_steps, config rate)
ALLOWED = [
    ("this schedule, at its end", 45000, 1.5e-05, 45000, 1.5e-05),
    ("this schedule, inside the warmup ramp", 45000, 3e-06, 45000, 1.5e-05),
    ("this schedule, rate logged as an int zero", 45000, 0, 45000, 1.5e-05),
]

REFUSED = [
    # The 30,000-step extension's own checkpoint, resumed by the 45,000 config: the
    # schedule that wrote it was shorter, so its scheduler.pt is not this one's.
    ("the previous, shorter schedule", 30000, 1.5e-05, 45000, 1.5e-05),
    ("the original 15,000-step run", 15000, 0.0, 45000, 1.5e-05),
    # A rate above the config's is the corruption the guard exists for: the original
    # peak restored from base_lrs is ten times the extension's rate.
    ("a rate ten times the config's", 45000, 1.5e-04, 45000, 1.5e-05),
    ("a longer schedule written over this one", 60000, 1.5e-05, 45000, 1.5e-05),
]


@pytest.mark.parametrize("name,ck_steps,ck_lr,cfg_steps,cfg_lr", ALLOWED,
                         ids=[c[0] for c in ALLOWED])
def test_segment_continuation_keeps_scheduler(guard_block, tmp_path, name,
                                              ck_steps, ck_lr, cfg_steps, cfg_lr):
    out = tmp_path / "run"
    make_checkpoint(out, 44000, ck_steps, ck_lr)
    r = run_guard(guard_block, out, 44000, cfg_steps, cfg_lr, tmp_path)
    assert r.returncode == 0, f"refused a segment continuation: {r.stderr}"
    assert "segment continuation" in r.stdout
    assert "REFUSING" not in r.stderr


@pytest.mark.parametrize("name,ck_steps,ck_lr,cfg_steps,cfg_lr", REFUSED,
                         ids=[c[0] for c in REFUSED])
def test_new_extension_still_refused(guard_block, tmp_path, name,
                                     ck_steps, ck_lr, cfg_steps, cfg_lr):
    out = tmp_path / "run"
    make_checkpoint(out, 30000, ck_steps, ck_lr)
    r = run_guard(guard_block, out, 30000, cfg_steps, cfg_lr, tmp_path)
    assert r.returncode == 1, f"allowed a resume it cannot vouch for: {r.stdout}"
    assert "REFUSING" in r.stderr
    # The refusal names both ways out and the cost of each. Removing the file
    # fixes the rate and loses the Adam moments, because transformers guards
    # both loads with one condition (measured 2026-11-18, job 363888), so the
    # message now points at re-pointing it first.
    assert "--scheduler rewrite" in r.stderr
    assert "moments" in r.stderr


def test_unreadable_state_is_refused(guard_block, tmp_path):
    """No trainer_state.json means the schedule cannot be read, so it is not vouched for."""
    out = tmp_path / "run"
    make_checkpoint(out, 30000, 45000, 1.5e-05, state=False)
    r = run_guard(guard_block, out, 30000, 45000, 1.5e-05, tmp_path)
    assert r.returncode == 1
    assert "REFUSING" in r.stderr


def test_empty_log_history_is_refused(guard_block, tmp_path):
    """A checkpoint that logged no rate cannot show which schedule it ran under."""
    out = tmp_path / "run"
    make_checkpoint(out, 30000, 45000, None)
    r = run_guard(guard_block, out, 30000, 45000, 1.5e-05, tmp_path)
    assert r.returncode == 1
    assert "REFUSING" in r.stderr


def test_guard_is_inside_the_extend_gate(guard_block):
    """With EXTEND unset the guard must not run at all."""
    text = (repo_root() / "workflows" / "bert-grid.sbatch").read_text(encoding="utf-8")
    gate = text.index('if [ "${EXTEND:-0}" = "1" ]; then')
    assert text.index(GUARD_START) > gate
    assert text.index("# --- end extension gate ---") > text.index(GUARD_START)


def test_no_scheduler_file_is_not_the_guards_business(guard_block, tmp_path):
    """A checkpoint with no scheduler.pt never reaches the decision."""
    out = tmp_path / "run"
    make_checkpoint(out, 30000, 30000, 1.5e-05, scheduler=False)
    r = run_guard(guard_block, out, 30000, 45000, 1.5e-05, tmp_path)
    assert r.returncode == 0
    assert r.stdout == ""


def write_repointed_scheduler(ck: Path, lr: float) -> None:
    """A scheduler.pt prepared the way scripts/prepare_rna_extension.py writes it."""
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "base_lrs": [lr, lr],
            "last_epoch": 0,
            "_step_count": 1,
            "_is_initial": True,
            "_get_lr_called_within_step": False,
            "_last_lr": [0.0, 0.0],
            "lr_lambdas": [{}, {}],
        },
        ck / "scheduler.pt",
    )


def test_a_repointed_scheduler_is_accepted(guard_block, tmp_path):
    """Prepared rather than amputated: the file stays, so optimizer.pt still loads.

    Removing scheduler.pt fixes the rate and loses the Adam moments, because
    transformers 4.45.1 guards both loads with one condition on its existence
    (trainer.py:3204) -- 42 optimizer state entries with the file present against 0
    without, measured 2026-11-18, job 363888. Re-pointing it keeps both.

    The checkpoint here was written under the OLD max_steps, so this is not the
    segment-continuation case and must not borrow its message.
    """
    out = tmp_path / "run"
    ck = make_checkpoint(out, 30000, 30000, 1.5e-05)
    write_repointed_scheduler(ck, 1.5e-06)
    r = run_guard(guard_block, out, 30000, 45000, 1.5e-06, tmp_path)
    assert r.returncode == 0, f"refused a prepared copy: {r.stderr}"
    assert "re-pointed at this config" in r.stdout
    assert "segment continuation" not in r.stdout, "printed both cases' messages"
    assert "REFUSING" not in r.stderr


def test_a_scheduler_left_at_the_old_peak_is_still_refused(guard_block, tmp_path):
    """last_epoch 0 is not enough; base_lrs has to be this config's rate."""
    out = tmp_path / "run"
    ck = make_checkpoint(out, 30000, 30000, 1.5e-05)
    write_repointed_scheduler(ck, 1.5e-05)
    r = run_guard(guard_block, out, 30000, 45000, 1.5e-06, tmp_path)
    assert r.returncode == 1, f"accepted the original peak: {r.stdout}"
    assert "REFUSING" in r.stderr


def test_an_unreadable_scheduler_refuses_rather_than_exiting_silently(guard_block, tmp_path):
    """Under set -e a bare assignment from a failing command substitution aborts.

    The rewrite check reads scheduler.pt with python and discards its stderr, so an
    unreadable file made the whole script exit 1 saying nothing at all and the
    refusal was never reached. Seen 2026-11-18 as an empty stderr in the extension
    gate's own test, after 64 links had already been queued on that code.
    """
    out = tmp_path / "run"
    make_checkpoint(out, 30000, 30000, 1.5e-05)  # scheduler.pt is written empty
    r = run_guard(guard_block, out, 30000, 45000, 1.5e-06, tmp_path, strict=True)
    assert r.returncode == 1
    assert "REFUSING" in r.stderr, f"exited without saying why: {r.stderr!r}"
