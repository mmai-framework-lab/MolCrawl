"""The eval subset stands in for the split, so for compounds it must not be its head.

The compounds sets are written in source-parquet order, whose leading rows are shorter
than the file as a whole, so a leading slice reports a loss below the split's own. It
must also be the same rows for every ladder size, which is why it is not drawn with the
run's training seed - each config carries a different one.

Off by default: protein, RNA and genome shuffle in prep, so their leading rows are
already a random sample and switching would only move the number by resampling noise.
"""
import pathlib

import pytest

from molcrawl.models.bert.main import (
    EVAL_SUBSET_ROWS,
    EVAL_SUBSET_SEED,
    subsample_eval_split,
)

CONFIG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "molcrawl/tasks/pretrain/configs"


class FakeSplit:
    """Stands in for a datasets.Dataset: shuffle(seed=...) then select(range(n))."""

    def __init__(self, rows, seed=None):
        self.rows, self.seed = list(rows), seed

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed):
        import random

        shuffled = list(self.rows)
        random.Random(seed).shuffle(shuffled)
        return FakeSplit(shuffled, seed)

    def select(self, rng):
        return FakeSplit([self.rows[i] for i in rng], self.seed)


def test_a_split_smaller_than_the_subset_is_returned_whole():
    split = FakeSplit(range(500))
    assert subsample_eval_split(split, random_sample=True) is split


def test_the_default_is_unchanged_for_every_other_modality():
    """Off by default, so a live protein or RNA investigation does not shift under it."""
    got = subsample_eval_split(FakeSplit(range(100000)))
    assert got.rows == list(range(EVAL_SUBSET_ROWS))


def test_opting_in_moves_off_the_leading_rows():
    """The defect being fixed: a source-ordered split's head is not representative."""
    got = subsample_eval_split(FakeSplit(range(100000)), random_sample=True)
    assert len(got) == EVAL_SUBSET_ROWS
    assert got.rows != list(range(EVAL_SUBSET_ROWS))


def test_the_same_rows_come_back_every_time():
    """Determinism is what keeps best_val free of resampling noise across eval points."""
    a = subsample_eval_split(FakeSplit(range(100000)), random_sample=True)
    b = subsample_eval_split(FakeSplit(range(100000)), random_sample=True)
    assert a.rows == b.rows


def test_ladder_sizes_share_the_rows_despite_differing_training_seeds():
    """Passing a per-config seed here would give each size a different eval set."""
    rows = [subsample_eval_split(FakeSplit(range(100000)), random_sample=True).rows for _ in range(3)]
    assert rows[0] == rows[1] == rows[2]
    # and an explicit override really does change them, so the knob is live
    other = subsample_eval_split(FakeSplit(range(100000)), random_sample=True, seed=EVAL_SUBSET_SEED + 1)
    assert other.rows != rows[0]


def test_only_compounds_bert_opts_in():
    """Pins the scope: turning it on elsewhere changes numbers for no gain."""
    on = {
        path.relative_to(CONFIG_ROOT).as_posix()
        for path in CONFIG_ROOT.rglob("*.py")
        if "eval_subset_random = True" in path.read_text(encoding="utf-8")
    }
    assert on == {
        "compounds/bert_small.py",
        "compounds/bert_medium.py",
        "compounds/bert_large.py",
    }, on


@pytest.mark.parametrize("modality", ["compounds", "protein_sequence"])
def test_training_seeds_differ_between_ladder_sizes(modality):
    """The premise of the test above, pinned against the configs."""
    seeds = set()
    for size in ("small", "medium", "large"):
        text = (CONFIG_ROOT / modality / f"bert_{size}.py").read_text(encoding="utf-8")
        seeds.update(
            line.split("=", 1)[1].strip() for line in text.splitlines() if line.startswith("seed =")
        )
    assert len(seeds) == 3, f"{modality}: expected one seed per size, got {seeds}"
