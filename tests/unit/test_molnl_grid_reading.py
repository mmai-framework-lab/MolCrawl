"""The two prose readings of the molnl batch grid: divergence, and which batch won.

The 2026-11-27 directive fixes a divergence rule for prose that differs from the one
the per-arm summary's ``collapsed`` column applies, and asks for both to be kept and
labelled. A rule that lives only in a sentence gets counted by hand and comes out
differently each time, so it lives in scripts/molnl_grid_reading.py and is held here.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import molnl_grid_reading as R  # noqa: E402


def points(*values):
    return [(i * 100, v) for i, v in enumerate(values)]


def test_a_point_at_the_floor_after_the_minimum_is_divergence():
    assert R.diverged(points(7.0, 4.0, 5.0, R.FLOOR["bert"]), R.FLOOR["bert"])


def test_a_curve_that_only_goes_down_is_not_divergence():
    assert not R.diverged(points(7.0, 3.0, 1.0, 0.5), R.FLOOR["bert"])


def test_rising_without_reaching_the_floor_is_not_divergence():
    """The rule asks for the floor, not merely for a rise off the minimum."""
    assert not R.diverged(points(7.0, 0.5, 2.0, 4.0), R.FLOOR["bert"])


def test_the_minimum_is_taken_at_its_first_occurrence():
    """A curve flat at its minimum has not turned at the last of those points."""
    flat = points(7.0, 0.5, 0.5, 0.5)
    assert not R.diverged(flat, R.FLOOR["bert"])
    turned = points(7.0, 0.5, 0.5, 9.0)
    assert R.diverged(turned, R.FLOOR["bert"])


def test_the_start_of_a_curve_is_above_the_floor_and_does_not_count():
    """Every BERT arm starts near 7, which is above the floor, and before the minimum."""
    assert not R.diverged(points(7.1, 6.0, 3.0, 0.1), R.FLOOR["bert"])


def test_each_architecture_is_judged_against_its_own_floor():
    """4.6500 is above the BERT floor and below the GPT-2 one, so it separates them."""
    between = points(7.0, 1.0, 4.65)
    assert R.diverged(between, R.FLOOR["bert"])
    assert not R.diverged(between, R.FLOOR["gpt2"])
    series = {("bert", "small", 256, "1e-3"): between,
              ("gpt2", "small", 256, "1e-3"): between}
    flagged = {k[0] for k, _, _ in R.divergence(series)}
    assert flagged == {"bert"}


def test_a_batch_is_worth_the_lowest_of_its_cells():
    """Each cell is one rate; the grid takes the best rate a batch reached."""
    series = {
        ("bert", "small", 256, "a"): points(7.0, 0.30),
        ("bert", "small", 256, "b"): points(7.0, 0.20),
        ("bert", "small", 512, "a"): points(7.0, 0.25),
    }
    row, = R.winners(series)
    assert (row["batch"], row["value"]) == (256, 0.20)
    assert (row["second_batch"], row["second_value"]) == (512, 0.25)
    assert abs(row["gap"] - 0.05) < 1e-12
    assert row["batches"] == 2


def test_the_gap_is_between_batches_not_between_rates_of_one_batch():
    """Two rates of the winning batch must not become the first and second place."""
    series = {
        ("bert", "small", 256, "a"): points(7.0, 0.10),
        ("bert", "small", 256, "b"): points(7.0, 0.11),
        ("bert", "small", 512, "a"): points(7.0, 0.80),
    }
    row, = R.winners(series)
    assert row["second_batch"] == 512
    assert abs(row["gap"] - 0.70) < 1e-12


def test_a_size_with_one_batch_has_no_gap():
    series = {("gpt2", "xl", 256, "a"): points(3.0, 0.5)}
    row, = R.winners(series)
    assert row["second_batch"] is None and row["gap"] is None


def test_the_reader_keeps_only_the_judged_valid_series(tmp_path):
    """Train rows, and the valid metrics that are not the judged one, are dropped."""
    path = tmp_path / "curves.tsv"
    cols = ["arch", "size", "batch", "lr", "step", "split", "metric", "value"]
    rows = [
        {"arch": "bert", "size": "small", "batch": "256", "lr": "1e-3", "step": "100",
         "split": "valid", "metric": "eval_loss_mask", "value": "0.5"},
        {"arch": "bert", "size": "small", "batch": "256", "lr": "1e-3", "step": "100",
         "split": "valid", "metric": "eval_loss_copy", "value": "9.9"},
        {"arch": "bert", "size": "small", "batch": "256", "lr": "1e-3", "step": "100",
         "split": "train", "metric": "train_loss_mask", "value": "0.4"},
        {"arch": "gpt2", "size": "small", "batch": "256", "lr": "6e-4", "step": "100",
         "split": "valid", "metric": "val_loss", "value": "0.6"},
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("#! report-tsv 1\n# source: synthetic\n")
        writer = csv.DictWriter(handle, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    series = R.read_curves(str(path))
    assert set(series) == {("bert", "small", 256, "1e-3"),
                           ("gpt2", "small", 256, "6e-4")}
    assert series[("bert", "small", 256, "1e-3")] == [(100, 0.5)]
