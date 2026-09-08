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

import pytest

torch = pytest.importorskip("torch")

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


# ---------------------------------------------------------------------------
# The draw has to be reproducible
#
# Three seeds are scored so the spread can be reported, and that only means
# anything if a seed names one draw. The collator's RNG path runs through
# torch's global generator, which a future change to the collator or to the
# batching could break without any test noticing -- the value would simply drift
# between runs and the reported spread would silently include it.
# ---------------------------------------------------------------------------


class _StubModel:
    """Returns fixed logits, so any difference comes from the masking draw."""

    def __init__(self, vocab):
        self.vocab = vocab
        g = torch.Generator().manual_seed(0)
        self._table = torch.randn(vocab, generator=g)

    def __call__(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        logits = self._table.view(1, 1, -1).expand(b, t, self.vocab).clone()
        return type("Out", (), {"logits": logits})()


class _Rows:
    """The slice interface score() uses, over fixed-length genome-like rows."""

    def __init__(self, n, length=32):
        self._rows = [[7] + [i % 4 for i in range(length - 2)] + [8] for _ in range(n)]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, sl):
        return {"input_ids": self._rows[sl]}


def _collator(mask_id=9, prob=0.2):
    """Masks with torch's global RNG, the way the real collator does."""
    def call(batch):
        ids = torch.tensor([b["input_ids"] for b in batch])
        labels = ids.clone()
        sel = torch.rand(ids.shape) < prob
        sel[:, 0] = False
        sel[:, -1] = False
        labels[~sel] = sg.IGNORE_INDEX
        ids = ids.clone()
        ids[sel] = mask_id
        return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
    return call


def test_one_seed_names_one_draw():
    """Same seed twice: identical masked count and identical loss."""
    model, rows, coll = _StubModel(10), _Rows(24), _collator()

    a = sg.score(model, coll, rows, 1026, 8, "cpu", 9)
    b = sg.score(model, coll, rows, 1026, 8, "cpu", 9)

    assert a["masked_positions"] == b["masked_positions"]
    assert a["eval_loss_mask"] == b["eval_loss_mask"]
    assert a["degenerate_baseline"] == b["degenerate_baseline"]


def test_different_seeds_are_different_draws():
    """Otherwise the three-seed spread would be measuring nothing."""
    model, rows, coll = _StubModel(10), _Rows(24), _collator()

    a = sg.score(model, coll, rows, 1026, 8, "cpu", 9)
    c = sg.score(model, coll, rows, 1028, 8, "cpu", 9)

    assert a["masked_positions"] != c["masked_positions"]


def test_the_baseline_is_taken_at_the_positions_that_were_masked():
    """It must follow the draw, not be a constant carried alongside it."""
    model, rows, coll = _StubModel(10), _Rows(24), _collator()

    a = sg.score(model, coll, rows, 1026, 8, "cpu", 9)

    assert sum(a["label_counts"].values()) == a["masked_positions"]
