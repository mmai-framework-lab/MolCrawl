"""The evaluation-mask wrapper's contract: same batch, same draw, RNG left as found."""

import torch

from molcrawl.models._collators import FixedEvalMaskCollator


class _DrawingCollator:
    """Stands in for an MLM collator: whatever it returns comes from the global RNG."""

    def __call__(self, features):
        return {"draw": torch.rand(4), "rows": len(features)}


def _batch(*rows):
    return [{"input_ids": list(row)} for row in rows]


def test_same_batch_draws_the_same_however_the_rng_has_moved():
    collator = FixedEvalMaskCollator(_DrawingCollator(), seed=42)
    batch = _batch([1, 2, 3, 4], [5, 6, 7, 8])

    first = collator(batch)["draw"]
    torch.rand(1000)          # what a run does between two evaluations
    torch.manual_seed(7)
    second = collator(batch)["draw"]

    assert torch.equal(first, second)


def test_different_batches_draw_differently():
    collator = FixedEvalMaskCollator(_DrawingCollator(), seed=42)

    one = collator(_batch([1, 2, 3, 4]))["draw"]
    other = collator(_batch([9, 9, 9, 9]))["draw"]

    assert not torch.equal(one, other)


def test_a_different_seed_draws_differently():
    batch = _batch([1, 2, 3, 4])

    one = FixedEvalMaskCollator(_DrawingCollator(), seed=42)(batch)["draw"]
    other = FixedEvalMaskCollator(_DrawingCollator(), seed=43)(batch)["draw"]

    assert not torch.equal(one, other)


def test_the_global_rng_is_where_it_was_before_the_call():
    """Training draws from the same generator, so the wrapper must not move it."""
    collator = FixedEvalMaskCollator(_DrawingCollator(), seed=42)

    torch.manual_seed(1234)
    expected = torch.rand(4)

    torch.manual_seed(1234)
    collator(_batch([1, 2, 3, 4]))
    after = torch.rand(4)

    assert torch.equal(expected, after)


def test_tensor_rows_key_the_same_as_list_rows():
    """The eval dataset yields either, depending on whether it went through Arrow."""
    collator = FixedEvalMaskCollator(_DrawingCollator(), seed=42)

    as_lists = collator(_batch([1, 2, 3, 4]))["draw"]
    as_tensors = collator([{"input_ids": torch.tensor([1, 2, 3, 4])}])["draw"]

    assert torch.equal(as_lists, as_tensors)
