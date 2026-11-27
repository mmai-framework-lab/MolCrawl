"""What the 2026-11-27b additions must not get wrong.

Three things here would be invisible until a run had already spent its budget: an
added arm that differs from its cell's existing arms in something other than the rate,
a grace period that waits a different amount of data at each batch, and a regeneration
that rewrites the 48 configs the grid is running under.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import compounds_batch_sweep_configs as configs  # noqa: E402
from scripts.compounds_batch_sweep_added_arms import (  # noqa: E402
    DEGENERATE_THRESHOLD,
    EXPECTED_ADDED_LR,
    EXPECTED_GRACE,
    GRACE_ROWS,
    added_arms,
    check,
    grace_for,
)
from scripts.compounds_batch_sweep_arms import arms as grid_arms  # noqa: E402
from scripts.compounds_batch_sweep_arms import warmup_for  # noqa: E402

CONFIG_DIR = ROOT / "molcrawl" / "tasks" / "pretrain" / "configs" / "compounds"

#: The measured per-cell positions, as the per-cell table carries them. Only the
#: columns the derivation reads are given.
MEASURED = [
    ("small", "256", 2, "lowest"), ("small", "512", 2, "lowest"),
    ("small", "1024", 2, "lowest"), ("small", "2048", 3, "lowest"),
    ("small", "4096", 3, "lowest"),
    ("medium", "256", 2, "highest"), ("medium", "512", 2, "highest"),
    ("medium", "1024", 2, "highest"), ("medium", "2048", 3, "highest"),
    ("medium", "4096", 3, "highest"),
    ("large", "256", 2, "lowest"), ("large", "512", 2, "lowest"),
    ("large", "1024", 2, "highest"), ("large", "2048", 3, "middle"),
    ("large", "4096", 3, "highest"),
    ("xl", "256", 2, "highest"), ("xl", "512", 2, "highest"),
    ("xl", "1024", 2, "highest"), ("xl", "2048", 3, "middle"),
    ("xl", "4096", 3, "middle"),
]


def cells(measured=MEASURED):
    return [{"size": size, "batch": batch, "lrs_tried": str(tried),
             "lr_position": position} for size, batch, tried, position in measured]


def grid():
    return [dict(row, batch=str(row["batch"])) for row in grid_arms("micro-order")]


def test_the_rates_are_the_ones_the_directive_printed():
    rows = added_arms(cells())
    assert len(rows) == 42
    by_cell = {}
    for row in rows:
        by_cell.setdefault((row["size"], int(row["batch"])), []).append(row["lr"])
    assert len(by_cell) == 14
    for key, printed in EXPECTED_ADDED_LR.items():
        assert [float(v) for v in by_cell[key]] == [float(v) for v in printed], key


def test_a_middle_cell_and_a_running_size_get_nothing():
    rows = added_arms(cells())
    assert not [r for r in rows if r["size"] == "xl"]
    assert not [r for r in rows if r["size"] == "large" and r["batch"] == 2048]


def test_the_grace_period_waits_over_the_same_blocks_at_every_batch():
    # The point of anchoring it to blocks is that an arm at batch 4096 does not get
    # 16 times as much data before the rule can fire as one at batch 256.
    for batch, want in EXPECTED_GRACE.items():
        assert grace_for(batch) == want
        assert abs(grace_for(batch) * batch - GRACE_ROWS) < batch
    # and at 10 % it lands on the same number as the warmup, which is also 10 %
    for row in added_arms(cells()):
        assert int(row["degenerate_grace_steps"]) == warmup_for("bert", row["steps"])


def test_an_added_arm_differs_from_its_cell_only_in_the_rate():
    existing = grid()
    rows = added_arms(cells())
    assert check(rows, cells(), existing) == 0
    for row in rows:
        sibling = next(r for r in existing
                       if r["arch"] == "bert" and r["size"] == row["size"]
                       and int(r["batch"]) == int(row["batch"]))
        for column in ("steps", "warmup_steps", "eval_interval", "save_interval",
                       "micro_batch", "grad_accum", "world_size", "eval_batch"):
            assert str(row[column]) == str(sibling[column]), (row["lr"], column)
        assert float(row["lr"]) != float(sibling["lr"])


def test_a_direction_the_directive_does_not_list_is_reported():
    # large b2048's best sat in the middle. If a later reading moved it to an end, the
    # cell would be extended without the directive having asked for it, so the
    # derivation has to refuse rather than add three arms.
    moved = [(size, batch, tried, "highest" if (size, batch) == ("large", "2048")
              else position) for size, batch, tried, position in MEASURED]
    rows = added_arms(cells(moved))
    assert len(rows) == 45
    assert check(rows, cells(moved), grid()) > 0


def test_regenerating_a_running_arm_reproduces_its_config_byte_for_byte():
    # The 48 arms are mid-run. A regeneration that changed a byte of their configs
    # would put the file out of step with the process reading it.
    existing = grid_arms("micro-order")
    checked = 0
    for row in existing:
        path = CONFIG_DIR / Path(row["config"]).name
        if not path.exists():
            continue
        siblings = configs.siblings_of(existing, row)
        assert configs.render(row, siblings) == path.read_text(encoding="utf-8"), path
        checked += 1
    assert checked == 68


def test_an_added_config_carries_the_new_threshold_and_grace_and_nothing_else_new():
    rows = added_arms(cells())
    row = next(r for r in rows if r["size"] == "small" and r["batch"] == 512)
    text = configs.render(row, [row])
    assert f"degenerate_loss_threshold = {DEGENERATE_THRESHOLD}" in text
    assert "degenerate_grace_steps = 22500" in text
    assert "11,520,000 blocks" in text
    assert "seed = 42" in text
    assert "judge_on = \"eval_loss_mask\"" in text
    assert "eval_subset_seed" not in text
    # a running arm's config names neither of the two
    old = next(r for r in grid_arms("micro-order")
               if r["arch"] == "bert" and r["size"] == "small" and r["batch"] == 512)
    old_text = configs.render(old, configs.siblings_of(grid_arms("micro-order"), old))
    assert "degenerate_grace_steps" not in old_text
    assert "degenerate_loss_threshold = 2.395" in old_text
