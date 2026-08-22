"""The repo carries two IGNORE_INDEX constants with different values.

`_mlm_diagnostics.IGNORE_INDEX` is -100, the value HF's MLM collator writes into
unscored label positions. `_collators.ambiguity_aware_collator.IGNORE_INDEX` is -1 and
belongs to the CLM path, where get_batch blanks ambiguous targets itself.

Reading MLM labels against the CLM constant makes every position look scored: the
neighbour look-up baselines then took -100 as their context and missed about 80% of
the time, and the degenerate blend indexed the unigram table with -100. Nothing warns
about it, so these pin the values and the intent.
"""
from molcrawl.models._collators.ambiguity_aware_collator import IGNORE_INDEX as CLM_IGNORE
from molcrawl.models.bert._mlm_diagnostics import IGNORE_INDEX as MLM_IGNORE


def test_the_two_conventions_are_what_their_callers_expect():
    assert MLM_IGNORE == -100, "HF MLM collators fill unscored labels with -100"
    assert CLM_IGNORE == -1, "the CLM path blanks its own targets with -1"


def test_they_are_not_interchangeable():
    """If these ever converge, the distinction this guards stops mattering - and the
    reader should find out from a failing test, not from a silently wrong baseline."""
    assert MLM_IGNORE != CLM_IGNORE


def test_mlm_labels_are_read_with_the_mlm_constant():
    """A label array filled the HF way must select only the scored positions."""
    import torch

    labels = torch.tensor([[-100, 7, -100, 9, -100]])
    assert int((labels != MLM_IGNORE).sum()) == 2
    # The CLM constant would count every position as scored, which is the defect.
    assert int((labels != CLM_IGNORE).sum()) == 5
