"""Why these correlations are taken inside a group and never across two.

Assembly count runs 41-112 for `eukaryote_matched` and 473-885 for
`global_random`. The ranges do not overlap, so pooled it is almost a label for
the group, and any group difference reappears as a correlation with it. That
result would say the two groups differ -- the question, not the answer. The
test below constructs exactly that trap and checks the code does not fall in.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "clinvar_group_correlations.py"
_spec = importlib.util.spec_from_file_location("_group_corr", _SRC)
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def _auroc(values):
    return {("bert", "probe", "21"): values}


def test_a_perfect_line_is_one():
    assert gc.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_a_monotone_curve_is_one_for_spearman_but_not_for_pearson():
    xs, ys = [1, 2, 3, 4], [1, 4, 9, 16]
    assert gc.spearman(xs, ys) == pytest.approx(1.0)
    assert gc.pearson(xs, ys) < 1.0


def test_ties_are_averaged_so_input_order_does_not_change_the_answer():
    assert gc.spearman([1, 1, 2, 3], [5, 5, 6, 7]) == \
        gc.spearman([1, 1, 3, 2], [5, 5, 7, 6])


def test_two_points_give_no_correlation():
    """Any two points lie on a line. mammal_centered is one subset, and a group
    that small is left out rather than reported as a perfect fit."""
    assert gc.pearson([1, 2], [5, 9]) is None


def test_a_separated_variable_does_not_become_a_correlation(capsys):
    """The trap, built on purpose.

    Inside each group the AUROC is flat against assembly count -- no
    relationship at all. Across the groups, `global_random` has both more
    assemblies and a higher AUROC, so pooling the twenty would show a strong
    positive correlation that is only the group difference wearing another name.
    """
    flat_but_separated = {}
    axis = {}
    for i in range(10):
        e, g = f"eukaryote_matched_random_seed{i}", f"global_random_seed{i}"
        axis[e], axis[g] = 50 + i, 500 + i           # ranges do not overlap
        flat_but_separated[e] = 0.70                 # flat inside each group
        flat_but_separated[g] = 0.75

    rows = gc.correlate(_auroc(flat_but_separated), {"assemblies": axis})

    assert len(rows) == 2                            # one per group, never pooled
    for row in rows:
        assert row["n"] == 10
        assert row["pearson"] is None                # flat: no variance in y
    pooled = gc.pearson([axis[s] for s in flat_but_separated],
                        [flat_but_separated[s] for s in flat_but_separated])
    assert pooled > 0.9                              # what pooling would have said


def test_mammal_centered_is_not_folded_into_either_group():
    """It is one subset. One point has no correlation, and putting it in a
    group of ten would move that group's number."""
    values = {f"global_random_seed{i}": 0.70 + i / 100 for i in range(10)}
    values["mammal_centered"] = 0.50
    axis = {s: 500 + i for i, s in enumerate(sorted(values))}

    rows = gc.correlate(_auroc(values), {"assemblies": axis})

    assert {r["group"] for r in rows} == set(gc.GROUPS)
    assert [r["n"] for r in rows if r["group"] == "global_random"] == [10]


def test_only_trained_runs_are_correlated(tmp_path):
    """The control and the untrained floor share the table but are not subsets,
    and have no assembly count to correlate against."""
    tsv = tmp_path / "r.tsv"
    tsv.write_text("kind\tsubset\tfamily\tarch\tmethod\tfold\tauroc\tcheckpoint_step\tnote\n"
                   "run\tglobal_random_seed1\tglobal_random\tbert\tprobe\t21\t0.70\t\t\n"
                   "model-free\t-\t-\tbert\tprobe\t21\t0.62\t\t\n"
                   "untrained\tuntrained_seed0\t-\tbert\tprobe\t21\t0.58\t\t\n")

    got = gc.read_auroc(str(tsv))

    assert got[("bert", "probe", "21")] == {"global_random_seed1": 0.70}


def test_the_assembly_count_is_rows_not_lines(tmp_path):
    """The CSV has a header; counting lines would add one assembly to every
    subset and shift the axis by a constant."""
    (tmp_path / "global_random_seed1.csv").write_text(
        "taxid,assembly_accession\n1,GCF_1\n2,GCF_2\n")

    assert gc.assembly_counts(str(tmp_path)) == {"global_random_seed1": 2}


def test_the_window_shape_comes_from_the_measured_scan(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps([{"subset": "global_random_seed1",
                                 "mean_segment_len": 3580.7, "loss_fraction": 0.117}]))

    seg, loss = gc.window_shape(str(path))

    assert seg == {"global_random_seed1": pytest.approx(3580.7)}
    assert loss == {"global_random_seed1": pytest.approx(0.117)}


def test_a_sign_disagreement_is_recorded_rather_than_resolved():
    """At n=10 a split between the linear and the monotone answer is itself the
    finding; picking one would hide it."""
    values = {f"global_random_seed{i}": v for i, v in
              enumerate([0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.30])}
    axis = {f"global_random_seed{i}": i for i in range(10)}

    row = [r for r in gc.correlate(_auroc(values), {"assemblies": axis})
           if r["group"] == "global_random"][0]

    assert row["signs_agree"] is False
    assert row["pearson"] < 0 < row["spearman"]


def test_a_column_that_is_flat_only_after_rounding_still_counts_as_flat():
    """Ten copies of 0.70 average to 0.6999999999999998, so the deviations are
    about 1e-17 and their sum of squares about 1e-33 -- truthy. Testing against
    zero let a correlation through that was built entirely out of rounding
    error. The test is against the scale of the values instead."""
    assert gc.pearson([1, 2, 3], [0.7, 0.7, 0.7]) is None
    assert gc.pearson(list(range(10)), [0.7] * 10) is None


def test_a_small_but_real_spread_is_still_a_correlation():
    """The guard must not throw away a genuine relationship just because the
    values sit close together -- AUROCs inside one group often do."""
    assert gc.pearson([1, 2, 3, 4], [1.0, 1.001, 1.002, 1.003]) == pytest.approx(1.0)
