"""Which five checkpoints the existing arms are measured again at.

The five are a choice, and a hand-picked set would be unreproducible, so the rule is
code and these are its cases: the anchors it names, what it does when two of them land
on the same checkpoint, and that it never returns a cluster when a spread is available.
The tie-breaks are pinned too -- an arm re-measured twice has to choose the same five.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    path = repo_root() / "scripts" / "compounds_reval_points.py"
    spec = importlib.util.spec_from_file_location("compounds_reval_points", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: What small's three segments actually hold.
SMALL = ([11700, 12100, 12200, 12300, 12900, 13000, 13500, 13800, 14000, 14500, 15000]
         + [21700, 22400, 24500, 25800, 26500, 27100, 27700, 28000, 29000, 30000]
         + [32400, 36000, 37000, 37900, 39800, 41000, 42000, 42800, 43000, 45000])


def found_from(steps):
    return [(s, "-base", f"/runs/arm/checkpoint-{s}") for s in steps]


def test_five_points_are_chosen(mod):
    judged = {s: 1.0 / (i + 1) for i, s in enumerate(SMALL)}
    assert len(mod.choose(found_from(SMALL), judged)) == 5


def test_the_earliest_and_the_last_are_always_in(mod):
    judged = {s: 1.0 for s in SMALL}
    chosen = mod.choose(found_from(SMALL), judged)
    assert min(SMALL) in chosen and max(SMALL) in chosen


def test_the_anchors_are_the_nearest_surviving(mod):
    judged = {s: 1.0 for s in SMALL}
    chosen = mod.choose(found_from(SMALL), judged)
    assert 15000 in chosen and 30000 in chosen


def test_an_anchor_with_no_exact_checkpoint_takes_the_nearest(mod):
    steps = [11000, 14800, 15300, 29000, 45000]
    chosen = mod.choose(found_from(steps), {s: 1.0 for s in steps})
    assert 14800 in chosen  # 200 away, against 15300's 300


def test_the_adopted_checkpoint_is_in(mod):
    judged = {s: 1.0 for s in SMALL}
    judged[37900] = 0.01
    chosen = mod.choose(found_from(SMALL), judged)
    assert chosen[37900].startswith("adopted")


def test_the_reason_is_recorded_for_every_point(mod):
    judged = {s: 1.0 for s in SMALL}
    judged[26500] = 0.01
    chosen = mod.choose(found_from(SMALL), judged)
    assert all(isinstance(v, str) and v for v in chosen.values())


def test_a_coinciding_adopted_point_does_not_cost_a_point(mod):
    # The lowest value sits on the last checkpoint, which the rule already took.
    judged = {s: 1.0 for s in SMALL}
    judged[45000] = 0.01
    chosen = mod.choose(found_from(SMALL), judged)
    assert len(chosen) == 5


def test_the_filled_point_lands_inside_the_widest_gap(mod):
    judged = {s: 1.0 for s in SMALL}
    judged[45000] = 0.01
    chosen = mod.choose(found_from(SMALL), judged)
    extra = [s for s, why in chosen.items() if why == "widest gap"]
    assert len(extra) == 1
    # 15,000-30,000 and 30,000-45,000 are both 15,000 wide; the tie goes to the
    # earlier gap, so the filled point sits in the first of them.
    assert 15000 < extra[0] < 30000


def test_fewer_surviving_than_five_returns_what_there_is(mod):
    steps = [15000, 30000, 45000]
    chosen = mod.choose(found_from(steps), {s: 1.0 for s in steps})
    assert sorted(chosen) == steps


def test_one_surviving_checkpoint_is_not_an_error(mod):
    chosen = mod.choose(found_from([45000]), {45000: 1.0})
    assert sorted(chosen) == [45000]


def test_the_choice_is_the_same_on_a_second_pass(mod):
    judged = {s: 1.0 for s in SMALL}
    judged[26500] = 0.01
    first = mod.choose(found_from(SMALL), judged)
    second = mod.choose(found_from(SMALL), judged)
    assert first == second


def test_a_tie_goes_to_the_lower_step(mod):
    # 14,900 and 15,100 are both 100 from the anchor, so the anchor takes 14,900.
    # 15,100 may still come in later as the gap filler, which is a different reason.
    steps = [10000, 14900, 15100, 30000, 45000]
    chosen = mod.choose(found_from(steps), {s: 1.0 for s in steps})
    assert chosen[14900] == "nearest 15,000"
    assert chosen.get(15100) != "nearest 15,000"


def test_no_judged_value_still_chooses_four(mod):
    # A size whose states carry no eval value at the checkpoint's own step.
    chosen = mod.choose(found_from(SMALL), {s: None for s in SMALL})
    assert len(chosen) == 5  # the gap filler makes the fifth
    assert not any(why.startswith("adopted") for why in chosen.values())


# --- reading the existing values -------------------------------------------------

def test_the_recorded_value_comes_from_the_checkpoints_own_step(mod, tmp_path):
    ckpt = tmp_path / "checkpoint-30000"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text(json.dumps({"log_history": [
        {"step": 29900, "eval_loss": 9.9, "eval_loss_mask": 9.8},
        {"step": 30000, "eval_loss": 0.0688, "eval_loss_mask": 0.2673},
    ]}), encoding="utf-8")
    assert mod.recorded(str(ckpt), 30000) == {"eval_loss": 0.0688,
                                              "eval_loss_mask": 0.2673}


def test_a_checkpoint_without_a_state_file_is_empty_not_an_error(mod, tmp_path):
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    assert mod.recorded(str(ckpt), 1) == {}


def test_a_state_without_the_step_is_empty(mod, tmp_path):
    ckpt = tmp_path / "checkpoint-500"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"step": 400, "eval_loss": 1.0}]}),
        encoding="utf-8")
    assert mod.recorded(str(ckpt), 500) == {}


# --- collecting the fixed-mask measurements --------------------------------------

def test_a_written_measurement_is_matched_by_its_checkpoint_path(mod, tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    ckpt = tmp_path / "checkpoint-45000"
    ckpt.mkdir()
    (records / "a.json").write_text(json.dumps({
        "checkpoint": str(ckpt),
        "metrics": {"eval_loss": 0.0701, "eval_loss_mask": 0.2688},
    }), encoding="utf-8")
    got = mod.collected(str(records))
    assert got[str(ckpt.resolve())] == {"eval_loss": 0.0701,
                                        "eval_loss_mask": 0.2688}


def test_no_records_directory_is_empty_not_an_error(mod, tmp_path):
    assert mod.collected(str(tmp_path / "nothing")) == {}
    assert mod.collected("") == {}


# --- the arms it covers ----------------------------------------------------------

def test_all_four_sizes_and_three_segments_are_covered(mod):
    assert sorted(mod.ARMS) == ["large", "medium", "small", "xl"]
    assert mod.SEGMENTS == ["", "-ext30k", "-ext45k"]


def test_the_checkpoint_scan_reads_all_three_segments(mod, tmp_path):
    for segment, step in (("", 15000), ("-ext30k", 30000), ("-ext45k", 45000)):
        ckpt = tmp_path / (mod.ARMS["small"] + segment) / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
    found = mod.checkpoints(str(tmp_path), "small")
    assert [s for s, _, _ in found] == [15000, 30000, 45000]
    assert [seg for _, seg, _ in found] == ["-base", "-ext30k", "-ext45k"]


def test_a_directory_that_is_not_a_checkpoint_is_skipped(mod, tmp_path):
    run = tmp_path / mod.ARMS["small"]
    (run / "checkpoint-15000").mkdir(parents=True)
    (run / "runs").mkdir()
    (run / "checkpoint-best").mkdir()
    found = mod.checkpoints(str(tmp_path), "small")
    assert [s for s, _, _ in found] == [15000]
