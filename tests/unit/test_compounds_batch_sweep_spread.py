"""What the equal-fraction spread table must not get wrong.

The table exists so the report does not move when it is regenerated on a later
day. Each test here pins one way a date could leak back into it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compounds_batch_sweep_spread import (  # noqa: E402
    best_up_to,
    excluded,
    rows_for,
    series,
)


def curve(size, batch, lr, points, metric="eval_loss_mask"):
    return [
        {
            "arch": "bert",
            "size": size,
            "batch": batch,
            "lr": lr,
            "step": str(step),
            "metric": metric,
            "value": f"{value}",
        }
        for step, value in points
    ]


def summary(size, batch, lr, steps_planned, best, final, ended="budget",
            steps_reached=None):
    return {
        "size": size,
        "batch": batch,
        "lr": lr,
        "steps_planned": str(steps_planned),
        "steps_reached": str(steps_planned if steps_reached is None else steps_reached),
        "best_value": f"{best}",
        "final_value": f"{final}",
        "ended": ended,
    }


def test_an_arm_is_read_at_a_fraction_of_its_own_budget():
    # two batches of one size, 100 and 10 planned steps: at 50 % of budget the
    # first is read at step 50 and the second at step 5, not both at step 50.
    curves = curve("small", "256", "1e-4", [(50, 0.5), (100, 0.4)])
    curves += curve("small", "4096", "1e-3", [(5, 0.7), (10, 0.6)])
    rows = rows_for(
        curves,
        [
            summary("small", "256", "1e-4", 100, 0.4, 0.4),
            summary("small", "4096", "1e-3", 10, 0.6, 0.6),
        ],
        fractions=(0.5,),
    )
    assert len(rows) == 1
    assert rows[0]["best_of_batches"] == "0.500000"
    assert rows[0]["worst_of_batches"] == "0.700000"
    assert rows[0]["spread"] == "0.200000"


def test_a_size_is_skipped_until_every_batch_has_reached_the_fraction():
    # batch 4096 has nothing past 10 % of its budget, so the size has no 50 % row:
    # a spread between a subset of the batches is not the spread between batches.
    curves = curve("large", "256", "1e-4", [(50, 0.5)])
    curves += curve("large", "4096", "1e-3", [(1, 0.7)])
    rows = rows_for(
        curves,
        [
            summary("large", "256", "1e-4", 100, 0.5, 0.5, ended="running",
                    steps_reached=50),
            summary("large", "4096", "1e-3", 10, 0.7, 0.7, ended="running",
                    steps_reached=1),
        ],
        fractions=(0.5,),
    )
    assert rows == []


def test_the_lowest_value_counts_not_the_last_one():
    # an arm that turned upward still contributes the minimum it reached, because
    # that is the quantity the grid compares.
    points = [(10, 0.9), (50, 0.4), (100, 1.8)]
    assert best_up_to(points, 100) == 0.4
    assert best_up_to(points, 50) == 0.4
    assert best_up_to(points, 10) == 0.9
    assert best_up_to(points, 5) is None


def test_a_collapse_stopped_arm_is_excluded():
    rows = [summary("small", "256", "3.16e-4", 450000, 0.073, 2.56, ended="collapse-stop")]
    assert excluded(rows) == {("small", "256", "3.16e-4")}


def test_a_climbing_arm_is_excluded_even_though_the_detector_did_not_fire():
    # 0.5548 -> 1.3169 is far above the minimum but below the degenerate threshold,
    # so nothing stopped it; it measures the rate, not the batch.
    rows = [summary("xl", "2048", "5.63e-05", 56250, 0.554842, 1.316936, ended="running")]
    assert excluded(rows) == {("xl", "2048", "5.63e-05")}


def test_a_healthy_running_arm_is_not_excluded():
    rows = [summary("large", "256", "3.32e-05", 450000, 0.06775, 0.067895, ended="running")]
    assert excluded(rows) == set()


def test_only_the_judged_metric_is_read():
    curves = curve("small", "256", "1e-4", [(50, 0.5)])
    curves += curve("small", "256", "1e-4", [(50, 0.01)], metric="eval_loss")
    got = series(curves)
    assert got == {("small", "256", "1e-4"): [(50, 0.5)]}


def test_the_best_arm_of_a_batch_represents_that_batch():
    # a batch holding two rates is represented by the better of them, so a bad
    # rate does not widen the spread the table is about.
    curves = curve("medium", "2048", "1e-4", [(50, 0.9)])
    curves += curve("medium", "2048", "3e-4", [(50, 0.5)])
    curves += curve("medium", "4096", "2e-4", [(50, 0.6)])
    rows = rows_for(
        curves,
        [
            summary("medium", "2048", "1e-4", 100, 0.9, 0.9),
            summary("medium", "2048", "3e-4", 100, 0.5, 0.5),
            summary("medium", "4096", "2e-4", 100, 0.6, 0.6),
        ],
        fractions=(0.5,),
    )
    assert len(rows) == 1
    assert rows[0]["best_of_batches"] == "0.500000"
    assert rows[0]["worst_of_batches"] == "0.600000"
    assert rows[0]["best_batch"] == "2048"
    assert rows[0]["arms_read"] == 3


def test_an_arm_short_of_the_fraction_does_not_fill_the_row_from_an_earlier_point():
    # the xl arms stood at 44 % of budget while holding values from 40 %. Reading
    # them on a 100 % row restated a 44 % measurement as a finished one.
    curves = curve("xl", "256", "1e-5", [(40, 0.5)])
    curves += curve("xl", "4096", "4e-5", [(40, 0.7)])
    rows = rows_for(
        curves,
        [
            summary("xl", "256", "1e-5", 100, 0.5, 0.5, ended="running",
                    steps_reached=44),
            summary("xl", "4096", "4e-5", 100, 0.7, 0.7, ended="running",
                    steps_reached=44),
        ],
        fractions=(0.4, 1.0),
    )
    assert [row["fraction_of_budget"] for row in rows] == ["0.40"]
