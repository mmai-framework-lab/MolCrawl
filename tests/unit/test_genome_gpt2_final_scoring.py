"""What the GPT-2 final scoring measures, and on which positions.

The margin this feeds is a difference between a model's loss and the loss a
context-free predictor pays. A difference is only meaningful when both terms
come from the same positions, and the campaign's two terms did not: the
baseline was taken on the first 10,000 valid rows while ``best_val`` came from
64,000 rows drawn with replacement, minimum kept. These pin the properties that
make the two terms comparable -- one ordered pass, labels counted where loss was
paid, and rows weighted by the positions they carry.
"""
import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "score_genome_gpt2_final.py"
_spec = importlib.util.spec_from_file_location("_score_genome_gpt2", _SRC)
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)

VOCAB = 10
N_ID = 4          # genome's only resolving ambiguous code


class _Rows:
    """The slice-and-column access `score` makes of a datasets split."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, s):
        return {"input_ids": self._rows[s]}


class _FlatModel:
    """Predicts the uniform distribution everywhere, so every scored position
    pays exactly ln(vocab) and the arithmetic is checkable by hand."""

    def __call__(self, x, y):
        b, t = x.shape
        return torch.zeros(b, t, VOCAB), None


def test_a_uniform_draw_pays_ln_of_the_alphabet():
    assert sg._entropy(Counter({0: 25, 1: 25, 2: 25, 3: 25})) == math.log(4)


def test_an_empty_count_reports_nothing_rather_than_zero():
    """No scored position is an absent measurement, not a baseline of zero."""
    assert sg._entropy(Counter()) is None


def test_the_first_base_of_a_row_is_never_a_label():
    """Scoring is next-token, so a row of n tokens carries n-1 positions and
    its opening base is only ever an input."""
    rows = _Rows([[2, 0, 0, 0], [2, 1, 1, 1]])

    _, counts, n_scored, _ = sg.score(_FlatModel(), rows, [], 2, "cpu")

    assert n_scored == 2 * 3
    assert counts[2] == 0            # 'G' opens both rows and labels neither
    assert counts[0] == 3 and counts[1] == 3


def test_ambiguous_labels_leave_both_the_loss_and_the_baseline():
    """N is excluded from the loss the run paid, so it must be excluded from
    the baseline too -- otherwise the margin subtracts two different things."""
    rows = _Rows([[0, N_ID, 1, 1]])

    loss, counts, n_scored, n_ambiguous = sg.score(
        _FlatModel(), rows, [N_ID], 2, "cpu")

    assert n_ambiguous == 1
    assert n_scored == 2
    assert N_ID not in counts
    assert loss == pytest.approx(math.log(VOCAB))


def test_rows_weigh_by_the_positions_they_carry_not_by_being_a_row():
    """A batch whose ambiguity mask removes more positions must not count for
    the same as a full one; a mean of per-batch means would make it."""
    rows = _Rows([[0, 0, 0, 0], [0, N_ID, N_ID, N_ID]])

    loss, _, n_scored, n_ambiguous = sg.score(
        _FlatModel(), rows, [N_ID], 1, "cpu")     # batch size 1: one row each

    assert (n_scored, n_ambiguous) == (3, 3)
    assert loss == pytest.approx(math.log(VOCAB))


def test_the_baseline_is_taken_where_the_loss_was_paid():
    """Same pass, same positions: the counts `score` returns are exactly the
    labels that entered the loss, so the two cannot drift apart."""
    rows = _Rows([[0, 0, 0, 1]])

    _, counts, n_scored, _ = sg.score(_FlatModel(), rows, [], 2, "cpu")

    assert sum(counts.values()) == n_scored
    assert sg._entropy(counts) == pytest.approx(
        -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3))
