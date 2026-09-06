"""Which rows the derived-dataset check actually compares.

The campaign reports valid and test numbers, so those splits are compared in
full: one row of the wrong provenance would change a reported value. train is
too large for that and gets a sample -- but a stride sample alone steps over a
local reordering, which is the one way datasets.map can go wrong, so contiguous
blocks go with it.
"""
import argparse
import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "verify_bert_from_gpt2.py"
_spec = importlib.util.spec_from_file_location("_verify_bert", _SRC)
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)


def _args(**kw):
    base = dict(sample=100000, blocks=100, block_size=1000, seed=42)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("split", ["valid", "test"])
def test_valid_and_test_are_compared_row_for_row(split):
    idx, how = verify._positions(split, 50_000, _args())

    assert idx == list(range(50_000))
    assert how == "all 50,000 rows"


def test_train_is_sampled_because_it_is_too_large_to_walk():
    idx, how = verify._positions("train", 46_000_000, _args())

    assert len(idx) < 46_000_000
    assert "stride" in how and "contiguous blocks of 1,000" in how


def test_the_sample_carries_whole_contiguous_runs_not_only_a_stride():
    """A stride alone leaves no two adjacent positions; blocks must show up."""
    idx, _ = verify._positions("train", 46_000_000, _args())

    adjacent = sum(1 for a, b in zip(idx, idx[1:]) if b - a == 1)
    assert adjacent >= 100 * (1000 - 1)


def test_positions_are_sorted_and_unique_so_a_block_cannot_double_count():
    idx, _ = verify._positions("train", 1_000_000, _args(sample=1000, blocks=50))

    assert idx == sorted(set(idx))


def test_a_split_smaller_than_the_sample_is_walked_in_full():
    """No point sampling 900 rows out of 900."""
    idx, how = verify._positions("train", 900, _args())

    assert idx == list(range(900))
    assert how == "all 900 rows"


def test_the_choice_of_blocks_is_reproducible():
    a, _ = verify._positions("train", 10_000_000, _args())
    b, _ = verify._positions("train", 10_000_000, _args())
    c, _ = verify._positions("train", 10_000_000, _args(seed=7))

    assert a == b
    assert a != c


def test_a_block_never_runs_off_the_end_of_the_split():
    idx, _ = verify._positions("train", 200_001, _args(sample=1000, blocks=200))

    assert max(idx) < 200_001
