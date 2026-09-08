"""What the final scoring reports, and why it is not one number.

The campaign compares subsets on eval_loss_mask. That is a mean over the
positions one random draw happened to mask, so a single value carries the draw
inside it: a gap between two subsets is only real if it is bigger than the gap
the draw alone produces. Three seeds are scored and the spread is reported
beside the mean so that comparison can be made rather than assumed.

The degenerate baseline is computed from the labels at exactly those masked
positions, in the same pass. It is what a model pays for predicting the marginal
and ignoring context, so "how far above its own degenerate solution" is a
per-subset question, not one number for the campaign.
"""
import importlib.util
import math
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "score_genome_bert_final.py"
_spec = importlib.util.spec_from_file_location("_score_genome", _SRC)
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)


def test_a_uniform_draw_pays_ln_of_the_alphabet():
    """Four bases in equal measure: nothing to exploit, so ln 4."""
    assert sg._entropy(Counter({0: 25, 1: 25, 2: 25, 3: 25})) == math.log(4)


def test_a_skewed_draw_pays_less_than_ln_four():
    """GC content below 50% makes the marginal predictor better than chance."""
    h = sg._entropy(Counter({0: 30, 1: 30, 2: 20, 3: 20}))

    assert h < math.log(4)


def test_the_baseline_moves_with_composition_which_is_why_it_is_per_subset():
    """A GC-poor split and a balanced one cannot share one baseline."""
    balanced = sg._entropy(Counter({0: 25, 1: 25, 2: 25, 3: 25}))
    skewed = sg._entropy(Counter({0: 31, 1: 31, 2: 19, 3: 19}))

    assert balanced - skewed > 0.02        # far above the 0.0011 run-to-run noise


def test_no_masked_positions_reports_nothing_rather_than_zero():
    """An empty draw is unmeasured, not a baseline of zero."""
    assert sg._entropy(Counter()) is None


def test_a_label_absent_from_the_draw_does_not_enter_the_entropy():
    """Zero counts must not contribute a log of zero."""
    h = sg._entropy(Counter({0: 50, 1: 50, 2: 0, 3: 0}))

    assert h == math.log(2)
