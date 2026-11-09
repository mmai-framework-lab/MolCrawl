"""Reading the logged learning rate around the head of a continued stage.

The case the extraction exists for is the one the configs describe ambiguously: a
stage whose predecessor ended at a constant rate, where a fresh scheduler still
begins its ramp at zero. The rows must separate the head step -- which belongs to
the previous stage -- from the ramp that follows it.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    path = repo_root() / "scripts" / "lr_at_stage_head_tsv.py"
    spec = importlib.util.spec_from_file_location("lr_at_stage_head_tsv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(root: Path, name: str, rates: dict, extra_steps=()):
    """A run whose furthest checkpoint carries `rates` as {step: learning_rate}."""
    run = root / name
    steps = sorted(set(rates) | set(extra_steps))
    for s in steps:
        (run / f"checkpoint-{s}").mkdir(parents=True, exist_ok=True)
    history = [{"step": s, "learning_rate": rates[s]} for s in sorted(rates)]
    state = {"global_step": max(steps), "log_history": history}
    (run / f"checkpoint-{max(steps)}" / "trainer_state.json").write_text(
        json.dumps(state), encoding="utf-8")
    return run


def read_tsv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def ramp_from_constant(head: int, constant: float, warmup: int = 500, step: int = 100):
    """What a fresh constant_with_warmup writes after resuming at `head`."""
    rates = {head: constant}  # the previous stage's last point
    s = head + step
    while s <= head + 1000:
        factor = min(1.0, (s - head) / warmup)
        rates[s] = constant * factor
        s += step
    return rates


def test_the_ramp_starts_from_zero_even_after_a_constant_stage(mod, tmp_path):
    run = write_run(tmp_path, "arm-ext45k", ramp_from_constant(30_000, 1e-4))
    rows = mod.rows_for(run, 30_000, 600)
    by_offset = {int(r["offset_from_head"]): float(r["learning_rate"]) for r in rows}
    assert by_offset[0] == pytest.approx(1e-4)       # the stage before this one
    assert by_offset[100] == pytest.approx(2e-5)     # a fifth of the way up
    assert by_offset[500] == pytest.approx(1e-4)     # the ramp is over
    assert by_offset[100] < by_offset[0]


def test_the_head_step_is_labelled_as_the_previous_stage(mod, tmp_path):
    run = write_run(tmp_path, "arm-ext45k", ramp_from_constant(30_000, 1e-4))
    rows = mod.rows_for(run, 30_000, 600)
    assert rows[0]["kind"] == "head (previous stage)"
    assert rows[0]["step"] == 30_000
    inside = [r for r in rows if r["kind"] == "inside the 500-step ramp"]
    assert [int(r["offset_from_head"]) for r in inside] == [100, 200, 300, 400, 500]


def test_a_head_after_a_decayed_stage_starts_from_zero_too(mod, tmp_path):
    rates = ramp_from_constant(15_000, 1e-4)
    rates[15_000] = 0.0  # stage 1 ends at exactly zero
    run = write_run(tmp_path, "arm-ext30k", rates)
    rows = mod.rows_for(run, 15_000, 600)
    by_offset = {int(r["offset_from_head"]): float(r["learning_rate"]) for r in rows}
    assert by_offset[0] == 0.0
    assert by_offset[100] == pytest.approx(2e-5)
    # Here the rate climbs away from nothing; the drop at 30,000 is the other case.
    assert by_offset[100] > by_offset[0]


def test_the_window_bounds_the_rows(mod, tmp_path):
    run = write_run(tmp_path, "arm-ext45k", ramp_from_constant(30_000, 1e-4))
    rows = mod.rows_for(run, 30_000, 200)
    offsets = [int(r["offset_from_head"]) for r in rows]
    assert offsets[:3] == [0, 100, 200]
    assert max(offsets) == 1000  # the last logged step is kept whatever the window


def test_the_last_logged_rate_is_kept_and_named(mod, tmp_path):
    run = write_run(tmp_path, "arm-ext45k", ramp_from_constant(30_000, 1e-4))
    rows = mod.rows_for(run, 30_000, 200)
    assert rows[-1]["kind"] == "last logged"
    assert rows[-1]["step"] == 31_000


def test_a_step_logged_twice_keeps_the_last_value(mod):
    history = [{"step": 100, "learning_rate": 1e-5}, {"step": 100, "learning_rate": 2e-5}]
    assert mod.rate_series(history) == {100: 2e-5}


def test_rows_without_a_rate_are_skipped(mod):
    history = [{"step": 100, "loss": 0.5}, {"step": 200, "learning_rate": 1e-5}]
    assert mod.rate_series(history) == {200: 1e-5}


def test_a_run_spec_needs_the_head_step(mod):
    with pytest.raises(SystemExit):
        mod.split_run("/somewhere/arm-ext45k")
    assert mod.split_run("/somewhere/arm-ext45k@30000") == (Path("/somewhere/arm-ext45k"), 30_000)


def test_the_furthest_checkpoint_is_read(mod, tmp_path):
    run = write_run(tmp_path, "arm-ext45k", ramp_from_constant(30_000, 1e-4),
                    extra_steps=(30_500,))
    rows = mod.rows_for(run, 30_000, 600)
    assert rows[0]["source"].endswith("checkpoint-31000/trainer_state.json")


def test_a_run_without_a_checkpoint_is_refused(mod, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        mod.rows_for(tmp_path / "empty", 30_000, 600)


def test_a_history_without_rates_is_refused(mod, tmp_path):
    run = tmp_path / "arm"
    (run / "checkpoint-100").mkdir(parents=True)
    (run / "checkpoint-100" / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"step": 100, "loss": 0.5}]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        mod.rows_for(run, 0, 600)


def test_the_tsv_is_written_without_a_bom_and_with_lf(mod, tmp_path):
    run = write_run(tmp_path, "arm-ext45k", ramp_from_constant(30_000, 1e-4))
    out = tmp_path / "out.tsv"
    mod.main(["--run", f"{run}@30000", "--window", "600", "--out", str(out)])
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    rows = read_tsv(out)
    assert [r["run"] for r in rows] == ["arm-ext45k"] * len(rows)
    assert rows[0]["stage_head"] == "30000"


def test_two_runs_are_written_in_the_order_given(mod, tmp_path):
    a = write_run(tmp_path, "arm-ext30k", ramp_from_constant(15_000, 1e-4))
    b = write_run(tmp_path, "arm-ext45k", ramp_from_constant(30_000, 1e-4))
    out = tmp_path / "out.tsv"
    mod.main(["--run", f"{a}@15000", "--run", f"{b}@30000", "--out", str(out)])
    names = [r["run"] for r in read_tsv(out)]
    assert names[0] == "arm-ext30k" and names[-1] == "arm-ext45k"
