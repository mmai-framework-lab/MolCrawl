"""The statistics behind the fold-wise ClinVar verdict.

Whether a model beat the shared-statistics baseline is a paired question -- both
score the same variants -- and whether one group of subsets sits above another is
a rank question over two sets of ten independent runs. Neither is answered by
comparing point estimates or ranges, so both tests are checked against
scipy here rather than trusted.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "clinvar_final_analysis.py"
_spec = importlib.util.spec_from_file_location("_clinvar_final", _SRC)
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)


def test_chr_y_is_not_a_fold():
    """29 variants cannot carry a fold."""
    assert fa.FOLDS == ["21", "22", "X"]
    assert "Y" not in fa.FOLDS


def test_mann_whitney_matches_scipy_on_separated_groups():
    st = pytest.importorskip("scipy.stats")
    x = [0.62, 0.59, 0.58, 0.58, 0.56, 0.56, 0.55, 0.54, 0.54, 0.53]
    y = [0.545, 0.544, 0.543, 0.542, 0.542, 0.541, 0.540, 0.538, 0.534, 0.530]

    got = fa.mann_whitney(x, y)
    ref = st.mannwhitneyu(x, y, alternative="two-sided")

    assert got["u"] == pytest.approx(ref.statistic)
    assert got["p_two_sided"] == pytest.approx(ref.pvalue, rel=0.15)


def test_mann_whitney_handles_ties():
    st = pytest.importorskip("scipy.stats")
    x = [1.0, 1.0, 2.0, 3.0]
    y = [1.0, 2.0, 2.0, 4.0]

    got = fa.mann_whitney(x, y)
    ref = st.mannwhitneyu(x, y, alternative="two-sided")

    assert got["u"] == pytest.approx(ref.statistic)


def test_the_effect_size_is_the_probability_one_group_exceeds_the_other():
    """Fully separated groups give 1.0; that is what the number means."""
    got = fa.mann_whitney([5, 6, 7], [1, 2, 3])

    assert got["prob_x_gt_y"] == pytest.approx(1.0)


def test_identical_groups_are_not_separated():
    got = fa.mann_whitney([1, 2, 3], [1, 2, 3])

    assert got["prob_x_gt_y"] == pytest.approx(0.5)
    assert got["p_two_sided"] > 0.5


def test_spearman_matches_scipy():
    st = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    x = rng.normal(size=21)
    y = 0.4 * x + rng.normal(size=21)

    _p, sp = fa.pearson_spearman(x, y)

    assert sp == pytest.approx(st.spearmanr(x, y).statistic)


def test_pearson_matches_numpy():
    rng = np.random.default_rng(11)
    x = rng.normal(size=21)
    y = 0.6 * x + rng.normal(size=21)

    p, _sp = fa.pearson_spearman(x, y)

    assert p == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_a_paired_difference_of_a_score_with_itself_is_zero():
    """The interval must contain zero when nothing differs."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + y * 0.5
    idx = rng.integers(0, 200, size=(200, 200))

    lo, hi = fa.paired_ci(y, s, s, idx)

    assert lo == 0.0 and hi == 0.0
