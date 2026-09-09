"""What the zero-shot baselines are, and why the composition one is not among them.

The composition baseline fits a pathogenicity rate to the labels. Zero-shot never
sees a label -- it only says how likely a base is at a position -- so the line it
has to clear must be drawn by something that also never sees a label. These two
are scored exactly the way the models are: reference minus variant, higher
meaning the reference was the likelier sequence.
"""
import importlib.util
import math
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "clinvar_zeroshot_baselines.py"
_spec = importlib.util.spec_from_file_location("_zs_baselines", _SRC)
zb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zb)


def _row(seq, ref, alt, sig="Benign", chrom="21"):
    var = list(seq)
    var[len(seq) // 2] = alt
    return {"reference_sequence": seq, "variant_sequence": "".join(var),
            "ref": ref, "alt": alt, "ClinicalSignificance": sig, "chrom": chrom}


def test_the_unigram_score_is_higher_when_the_reference_base_is_commoner():
    """Same orientation as the models: reference likelier => higher score."""
    seq = "A" * 90 + "C" * 10
    rows = [_row(seq, "A", "C"), _row(seq, "C", "A")]

    scores, freq = zb.unigram_scores(rows)

    assert freq["A"] > freq["C"]
    assert scores[0] > 0 > scores[1]


def test_a_base_the_window_never_shows_scores_zero_rather_than_infinity():
    rows = [_row("ACGT" * 25, "A", "N")]

    scores, _ = zb.unigram_scores(rows)

    assert scores[0] == 0.0


def test_the_markov_model_learns_the_repeat_it_was_fitted_on():
    """After ACGTACGT..., G must be far likelier than A following AC."""
    rows = [_row("ACGT" * 64, "A", "C")]
    ctx, nxt, order = zb.fit_markov(rows, 3)

    p_g = (nxt[("AC", "G")] + 1.0) / (ctx["AC"] + 4.0)
    p_a = (nxt[("AC", "A")] + 1.0) / (ctx["AC"] + 4.0)

    assert p_g > 10 * p_a


def test_an_unseen_context_is_uninformative_not_impossible():
    """add-one keeps a novel context at 1/4 instead of a log of zero."""
    ctx, nxt, order = zb.fit_markov([_row("ACGT" * 64, "A", "C")], 3)

    ll = zb.markov_loglik("TTTTT", ctx, nxt, order)

    assert ll == pytest.approx(3 * math.log(0.25))


def test_the_difference_ignores_whatever_the_two_windows_share():
    """Both sequences are scored over the same positions, so shared context
    contributes the same amount to each and cancels in the difference.

    Extending both with an identical prefix must therefore leave the difference
    exactly where it was. If the two were scored over different ranges, the
    shared part would not cancel and the score would drift with window length.
    """
    rows = [_row("ACGT" * 64, "A", "T")]
    ctx, nxt, order = zb.fit_markov(rows, 5)
    r = rows[0]

    d_short = (zb.markov_loglik(r["reference_sequence"], ctx, nxt, order)
               - zb.markov_loglik(r["variant_sequence"], ctx, nxt, order))
    pad = "ACGT" * 16
    d_long = (zb.markov_loglik(pad + r["reference_sequence"], ctx, nxt, order)
              - zb.markov_loglik(pad + r["variant_sequence"], ctx, nxt, order))

    assert d_long == pytest.approx(d_short)


def test_a_substitution_changes_only_its_own_neighbourhood():
    """An order-4 model reads 5 positions around the substituted base.

    A base further away than that cannot move the likelihood, which is what makes
    the score local rather than a property of the whole window.
    """
    rows = [_row("ACGT" * 64, "A", "T")]
    ctx, nxt, order = zb.fit_markov(rows, 5)
    seq = rows[0]["reference_sequence"]

    near = list(seq)
    near[128] = "T"
    far = list(seq)
    far[128] = "T"
    far[200] = "T"

    d_near = zb.markov_loglik(seq, ctx, nxt, order) - zb.markov_loglik("".join(near), ctx, nxt, order)
    d_far = zb.markov_loglik(seq, ctx, nxt, order) - zb.markov_loglik("".join(far), ctx, nxt, order)

    assert d_far > d_near          # the second substitution costs more, on its own


def test_auroc_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y * 0.4

    assert zb.auroc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
