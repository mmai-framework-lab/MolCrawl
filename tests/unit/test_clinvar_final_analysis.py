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


# ---------------------------------------------------------------------------
# What the run is measured against
#
# The zero-shot comparison is against the 5-mer Markov model, written under
# "markov"; the linear probe's comparison is against its model-free control,
# written under "score". Same arithmetic, different file, so the field is an
# argument rather than a literal -- and a wrong name has to fail loudly, because
# silently reading the other baseline would change every verdict in the output.
# ---------------------------------------------------------------------------


def test_the_baseline_field_is_not_hardcoded():
    import inspect
    src = inspect.getsource(fa.main)

    assert '"--baseline-key"' in src
    assert 'base[v][key]' in src


def test_a_missing_baseline_field_stops_the_run(tmp_path, capsys):
    import json
    import pytest as _pytest
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps({"vcv_id": "V1", "chrom": "21",
                             "label_pathogenic": 1, "score": 0.5}) + "\n")
    scores = tmp_path / "s"
    (scores / "run").mkdir(parents=True)
    (scores / "run" / "predictions.jsonl").write_text(
        json.dumps({"vcv_id": "V1", "score": 0.5}) + "\n")

    import sys
    argv = ["x", "--scores-dir", str(scores), "--baseline-scores", str(p),
            "--baseline-key", "markov"]
    old = sys.argv
    sys.argv = argv
    try:
        with _pytest.raises(SystemExit) as e:
            fa.main()
    finally:
        sys.argv = old

    assert "markov" in str(e.value)


# ---------------------------------------------------------------------------
# The second correlation's x-axis is a margin -- a model's loss below the loss a
# context-free predictor pays. Both terms have to come from the same positions,
# or the range difference between them rides along on the x-axis and cannot be
# told from a real difference in how well the corpus was learned. --final-scores
# supplies terms measured in one pass; the older pair of inputs does not, and
# the output has to say which was used.
# ---------------------------------------------------------------------------


def _tiny_campaign(tmp_path, subsets):
    """Two subsets scored over the three folds, enough for a correlation."""
    import json
    rows, scores = [], tmp_path / "s"
    for i, (chrom, label) in enumerate(
            [("21", 1), ("21", 0), ("22", 1), ("22", 0), ("X", 1), ("X", 0)] * 2):
        rows.append({"vcv_id": f"V{i}", "chrom": chrom,
                     "label_pathogenic": label, "score": 0.1 * (i % 5)})
    b = tmp_path / "b.jsonl"
    b.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for n, (name, shift) in enumerate(subsets.items()):
        d = scores / name
        d.mkdir(parents=True)
        d.joinpath("predictions.jsonl").write_text("\n".join(
            json.dumps({"vcv_id": r["vcv_id"],
                        "score": r["score"] + shift * r["label_pathogenic"]})
            for r in rows) + "\n")
    return scores, b


def _run(argv):
    import json
    import sys
    old = sys.argv
    sys.argv = ["x"] + argv
    try:
        fa.main()
    finally:
        sys.argv = old
    return json.load(open(argv[argv.index("--out") + 1]))


def test_a_matched_margin_is_read_and_its_range_recorded(tmp_path):
    """The GPT-2 scorer writes margin as the number itself."""
    import json
    subsets = {"global_random_seed1": 0.3, "eukaryote_matched_random_seed1": 0.1}
    scores, b = _tiny_campaign(tmp_path, subsets)
    fs = tmp_path / "final"
    fs.mkdir()
    for name, m in zip(subsets, (0.28, 0.24)):
        fs.joinpath(f"{name}.json").write_text(json.dumps({"margin": m}))

    out = _run(["--scores-dir", str(scores), "--baseline-scores", str(b),
                "--baseline-key", "score", "--rounds", "20",
                "--final-scores", str(fs), "--out", str(tmp_path / "o.json")])

    src = out["correlations"]["退化解からの余裕"]["source"]
    assert src["from"] == "final-scores" and src["n"] == 2


def test_the_bert_scorers_seed_aggregate_is_read_as_its_mean(tmp_path):
    """The BERT scorer averages over masking draws and writes a dict; the
    GPT-2 pass is deterministic and writes a number. Both are margins."""
    import json
    subsets = {"global_random_seed1": 0.3, "eukaryote_matched_random_seed1": 0.1}
    scores, b = _tiny_campaign(tmp_path, subsets)
    fs = tmp_path / "final"
    fs.mkdir()
    for name, m in zip(subsets, (0.28, 0.24)):
        fs.joinpath(f"{name}.json").write_text(json.dumps(
            {"margin": {"mean": m, "min": m - 0.001, "max": m + 0.001,
                        "spread": 0.002}}))

    out = _run(["--scores-dir", str(scores), "--baseline-scores", str(b),
                "--baseline-key", "score", "--rounds", "20",
                "--final-scores", str(fs), "--out", str(tmp_path / "o.json")])

    assert out["correlations"]["退化解からの余裕"]["n"] == 2


def test_the_unmatched_pair_is_marked_unmatched_rather_than_refused(tmp_path):
    """Earlier figures still have to be readable, but the output must carry
    the fact that its two terms came from different ranges."""
    import json
    subsets = {"global_random_seed1": 0.3, "eukaryote_matched_random_seed1": 0.1}
    scores, b = _tiny_campaign(tmp_path, subsets)
    deg = tmp_path / "deg.json"
    deg.write_text(json.dumps([{"subset": n, "gpt2": {"baseline": 1.37}}
                               for n in subsets]))
    csvp = tmp_path / "per-run.csv"
    csvp.write_text("subset,best_val\n" + "".join(
        f"{n},{v}\n" for n, v in zip(subsets, (1.09, 1.13))))

    out = _run(["--scores-dir", str(scores), "--baseline-scores", str(b),
                "--baseline-key", "score", "--rounds", "20",
                "--degenerate-baselines", str(deg), "--per-run-csv", str(csvp),
                "--out", str(tmp_path / "o.json")])

    assert out["correlations"]["退化解からの余裕"]["source"]["matched"] is False


def test_final_scores_wins_over_the_unmatched_pair_when_both_are_given(tmp_path):
    """A driver that passes the old inputs out of habit must not silently get
    the unmatched margin once the matched one exists."""
    import json
    subsets = {"global_random_seed1": 0.3, "eukaryote_matched_random_seed1": 0.1}
    scores, b = _tiny_campaign(tmp_path, subsets)
    fs = tmp_path / "final"
    fs.mkdir()
    for name, m in zip(subsets, (0.28, 0.24)):
        fs.joinpath(f"{name}.json").write_text(json.dumps({"margin": m}))
    deg = tmp_path / "deg.json"
    deg.write_text(json.dumps([{"subset": n, "gpt2": {"baseline": 1.37}}
                               for n in subsets]))
    csvp = tmp_path / "per-run.csv"
    csvp.write_text("subset,best_val\n" + "".join(
        f"{n},{v}\n" for n, v in zip(subsets, (1.09, 1.13))))

    out = _run(["--scores-dir", str(scores), "--baseline-scores", str(b),
                "--baseline-key", "score", "--rounds", "20",
                "--final-scores", str(fs),
                "--degenerate-baselines", str(deg), "--per-run-csv", str(csvp),
                "--out", str(tmp_path / "o.json")])

    assert out["correlations"]["退化解からの余裕"]["source"]["from"] == "final-scores"
