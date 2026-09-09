"""What the ClinVar aggregation computes, and why it is paired.

Every run scores the same variants, so subsets can be compared variant for
variant. That matters on the held-out chromosomes: a balanced comparison of
1,646 variants gives an accuracy interval of about ±0.024, far wider than the
differences between subsets, while a paired difference is bounded by how much
two runs disagree on the same variant rather than by the size of the test set.

The intervals are bootstrapped over variants. There is one run per subset, so
run-to-run spread is not measured and cannot be claimed; overlapping intervals
mean two subsets are not ordered by this evidence.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "clinvar_aggregate.py"
_spec = importlib.util.spec_from_file_location("_clinvar_aggregate", _SRC)
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)


def test_a_perfect_ranking_scores_one():
    assert ca.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_a_reversed_ranking_scores_zero():
    assert ca.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_all_scores_equal_is_exactly_chance():
    """Ties must average, or a constant predictor reads as 1.0 or 0.0."""
    assert ca.auroc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_ties_are_averaged_not_broken_by_input_order():
    a = ca.auroc([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1])
    b = ca.auroc([0, 1, 0, 1], [0.5, 0.5, 0.1, 0.9])

    assert a == b


def test_one_class_only_is_undefined_rather_than_a_number():
    """An AUROC over a single class is not zero or one; it does not exist."""
    assert np.isnan(ca.auroc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_it_matches_sklearn_on_random_data():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.normal(size=500) + y * 0.3

    assert ca.auroc(y, s) == pytest.approx(sk.roc_auc_score(y, s))


# ---------------------------------------------------------------------------
# The degenerate baseline
#
# A model that has learned nothing about context still scores above 0.5 when
# pathogenic and benign variants differ in which bases they involve. The line a
# result has to clear is therefore not 0.5, and it is per dataset.
# ---------------------------------------------------------------------------


def _rows(pairs, labels):
    return [{"ref_allele": r, "alt_allele": a, "label_pathogenic": y}
            for (r, a), y in zip(pairs, labels)]


def test_base_composition_alone_can_beat_chance():
    """C>T pathogenic, A>G benign: nothing about context, and yet separable."""
    rows = _rows([("C", "T")] * 50 + [("A", "G")] * 50, [1] * 50 + [0] * 50)

    assert ca.composition_baseline(rows) > 0.9


def test_composition_carries_no_signal_when_the_pairs_do_not_differ():
    rows = _rows([("C", "T")] * 100, [1] * 50 + [0] * 50)

    assert ca.composition_baseline(rows) == pytest.approx(0.5)


def test_the_bootstrap_is_reproducible():
    a = ca.boot_indices(100, 10, 1026)
    b = ca.boot_indices(100, 10, 1026)
    c = ca.boot_indices(100, 10, 1027)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
