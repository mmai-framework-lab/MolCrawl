"""What the per-cell and per-arm tables must not get wrong.

Two claims are easy to make by accident here: that the winning rate sat at the
edge of the swept range when a collapsed arm above it was quietly dropped, and
that an arm's level is its series minimum when the minimum is one excursion out
of a flat band.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compounds_batch_sweep_cells import (  # noqa: E402
    arm_table_rows,
    band,
    cell_rows,
    margins,
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
            steps_reached=None, stopped_at=""):
    return {
        "arch": "bert",
        "size": size,
        "batch": batch,
        "lr": lr,
        "steps_planned": str(steps_planned),
        "steps_reached": str(steps_planned if steps_reached is None else steps_reached),
        "best_value": f"{best}",
        "final_value": f"{final}",
        "ended": ended,
        "stopped_at_value": stopped_at,
    }


def arm(size, batch, lr, multiplier):
    return {"arch": "bert", "size": size, "batch": batch, "lr": lr,
            "lr_multiplier": multiplier}


def test_the_rate_rank_counts_collapsed_arms_too():
    # Three rates were swept and the top one collapsed. The winner is the middle
    # rate, not the highest: dropping the collapsed arm would claim the cell's
    # optimum might lie above the swept range when the range went higher.
    summaries = [
        summary("small", "256", "1e-4", 100, 0.5, 0.5),
        summary("small", "256", "2e-4", 100, 0.4, 0.4),
        summary("small", "256", "3e-4", 100, 1.0, 2.8, ended="collapse-stop",
                steps_reached=70, stopped_at="2.7322"),
    ]
    curves = (curve("small", "256", "1e-4", [(100, 0.5)])
              + curve("small", "256", "2e-4", [(100, 0.4)])
              + curve("small", "256", "3e-4", [(70, 1.0)]))
    arms = [arm("small", "256", "1e-4", "0.7"), arm("small", "256", "2e-4", "1.0"),
            arm("small", "256", "3e-4", "1.4")]
    (row,) = cell_rows(arms, summaries, curves, "src")
    assert row["lrs_tried"] == 3
    assert row["best_lr"] == "2e-4"
    assert row["best_lr_rank"] == 2
    assert row["lr_position"] == "middle"
    assert row["arms_surviving"] == 2
    assert row["arms_collapsed"] == 1


def test_the_only_surviving_rate_is_reported_as_the_lowest():
    # The cell's higher rate collapsed, so the survivor is the bottom of the
    # swept range and the cell's optimum may lie below it.
    summaries = [
        summary("small", "512", "1e-4", 100, 0.5, 0.5),
        summary("small", "512", "2e-4", 100, 1.0, 3.1, ended="collapse-stop",
                steps_reached=50),
    ]
    curves = (curve("small", "512", "1e-4", [(100, 0.5)])
              + curve("small", "512", "2e-4", [(50, 1.0)]))
    (row,) = cell_rows([], summaries, curves, "src")
    assert row["lr_position"] == "lowest"
    assert row["arms_surviving"] == 1


def test_a_never_learning_arm_is_reported_by_its_band_not_its_minimum():
    # One excursion to 0.5 inside a band that otherwise sits near 1.3. The arm is
    # still running and far above its own minimum, so the band represents it.
    points = [(10, 2.2), (20, 1.8)] + [(30, 0.5)] + [
        (step, 1.3) for step in range(40, 100, 10)
    ]
    summaries = [summary("xl", "2048", "5e-5", 100, 0.5, 1.3, ended="running",
                         steps_reached=90)]
    curves = curve("xl", "2048", "5e-5", points)
    (row,) = arm_table_rows([], summaries, curves, "src")
    assert row["mark"] == "not-learning"
    assert row["value_kind"] == "band_mean"
    # The band starts at 20 % of the budget, inclusive: the 1.8 at step 20 is the
    # first point in it and the 2.2 at step 10 is left out. The excursion to 0.5 at
    # step 30 is inside the band, so it widens the ends without moving the mean to
    # where the minimum sits.
    assert float(row["band_high"]) == 1.8
    assert float(row["band_low"]) == 0.5
    assert int(row["band_points"]) == 8
    assert float(row["value"]) > 1.0


def test_a_converged_arm_keeps_its_minimum():
    summaries = [summary("large", "4096", "2.66e-4", 100, 0.0665, 0.0665)]
    curves = curve("large", "4096", "2.66e-4", [(50, 0.07), (100, 0.0665)])
    (row,) = arm_table_rows([], summaries, curves, "src")
    assert row["mark"] == ""
    assert row["value_kind"] == "minimum"
    assert row["value"] == "0.066500"


def test_a_collapsed_arm_is_marked_and_keeps_its_stopping_value():
    summaries = [summary("xl", "4096", "7.97e-5", 100, 0.4998, 2.6643,
                         ended="collapse-stop", steps_reached=22,
                         stopped_at="2.664300")]
    curves = curve("xl", "4096", "7.97e-5", [(10, 0.4998), (22, 2.6643)])
    (row,) = arm_table_rows([], summaries, curves, "src")
    assert row["mark"] == "collapsed"
    assert row["stopped_at_value"] == "2.664300"
    # a collapsed arm did reach a minimum before it turned, and that is the value
    # the arm-level inventory carries
    assert row["value_kind"] == "minimum"


def test_the_band_is_empty_before_the_arm_reaches_it():
    assert band([(5, 1.0)], 100) is None
    assert band([(5, 1.0), (50, 1.2)], 100)["points"] == 1


def test_the_margin_is_between_the_two_best_batches():
    cells = [
        {"size": "small", "batch": "256", "value": "0.067191"},
        {"size": "small", "batch": "1024", "value": "0.066936"},
        {"size": "small", "batch": "2048", "value": "0.066943"},
    ]
    first, second, gap = margins(cells)["small"]
    assert first[1] == "1024"
    assert second[1] == "2048"
    assert abs(gap - 0.000007) < 1e-9
