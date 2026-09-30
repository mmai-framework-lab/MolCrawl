"""What the results table and the architecture pairing have to get right.

Both read files written by four different scripts over two months, and both
feed numbers straight into a report. The failures pinned here are the ones that
produce a plausible table rather than an error: a fold that silently vanishes
because two files spell it differently, a range quoted over a family of one,
and a paired test that is not actually paired.
"""
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_ROOT = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", _ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = _load("clinvar_results_table")
ap = _load("clinvar_arch_pairs")


# --- the table ------------------------------------------------------------

def test_a_fold_is_found_under_either_spelling():
    """fold-analysis.json says 21, probe.json says chr21. A lookup that knows
    only one spelling drops every fold of the other file without erroring --
    the table simply comes out short and nothing says so."""
    assert rt.fold_key("21", {"21", "22", "X"}) == "21"
    assert rt.fold_key("21", {"chr21", "chr22", "chrX"}) == "chr21"
    assert rt.fold_key("21", {"chrY"}) is None


def test_chr_y_is_not_a_fold():
    assert "Y" not in rt.FOLDS and "chrY" not in rt.FOLDS


@pytest.mark.parametrize("subset,family", [
    ("mammal_centered", "mammal_centered"),
    ("eukaryote_matched_random_seed7", "eukaryote_matched"),
    ("global_random_seed10", "global_random"),
])
def test_a_subset_lands_in_its_corpus_family(subset, family):
    assert rt.family_of(subset) == family


def test_a_family_of_one_gets_no_range():
    """mammal_centered is one run. A range over one value is not a measure of
    anything, and printed beside two real ranges it reads as one."""
    rows = [["run", "mammal_centered", "mammal_centered", "bert", "probe", "21", "0.80", "", ""],
            ["run", "global_random_seed1", "global_random", "bert", "probe", "21", "0.70", "", ""],
            ["run", "global_random_seed2", "global_random", "bert", "probe", "21", "0.76", "", ""]]

    summary = {(s["family"], s["fold"]): s for s in rt.summarise(rows)}

    assert summary[("mammal_centered", "21")]["range"] is None
    assert summary[("global_random", "21")]["range"] == pytest.approx(0.06)


def test_only_trained_runs_enter_the_group_summary():
    """The control and the floor are rows of the same table so they travel with
    it, but averaging them into a family would move the family's number."""
    rows = [["run", "global_random_seed1", "global_random", "bert", "probe", "21", "0.70", "", ""],
            ["model-free", "-", "-", "bert", "probe", "21", "0.62", "", ""],
            ["untrained", "untrained_seed0", "-", "bert", "probe", "21", "0.58", "", ""]]

    summary = rt.summarise(rows)

    assert len(summary) == 1 and summary[0]["mean"] == pytest.approx(0.70)


def test_the_checkpoint_step_comes_from_one_place_for_both_methods(tmp_path):
    """Zero-shot and the probe score the checkpoint the run itself selected, so
    there is one step per run. Reading it from two files would let them
    disagree on the same run."""
    analysis = {"runs": {"global_random_seed1": {"per_fold": {"21": {"auroc": 0.7}}}}}
    probe = {"runs": {"global_random_seed1": {"checkpoint": "checkpoint-139000"}}}

    zero = rt.run_rows(analysis, probe, "bert", "zeroshot")
    prob = rt.run_rows(analysis, probe, "bert", "probe")

    assert zero[0][7] == "139000" and prob[0][7] == "139000"


def test_a_nanogpt_checkpoint_name_carries_no_step(tmp_path):
    """GPT-2 adopts ckpt.pt, which has no step in its name. The field is left
    as it is rather than invented."""
    analysis = {"runs": {"mammal_centered": {"per_fold": {"21": {"auroc": 0.7}}}}}
    probe = {"runs": {"mammal_centered": {"checkpoint": "ckpt.pt"}}}

    assert rt.run_rows(analysis, probe, "gpt2", "probe")[0][7] == "ckpt.pt"


# --- the pairing ----------------------------------------------------------

def test_a_perfect_ranking_scores_one():
    assert ap.auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)


def test_ties_are_averaged_not_ordered_by_position():
    """All-equal scores carry no ranking, so the answer is 0.5 whichever way
    the ties happen to be laid out."""
    assert ap.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_the_bootstrap_draws_one_index_set_for_both_sides():
    """The two sides scored the same variants. Resampling them separately would
    throw the pairing away and widen every interval."""
    import inspect
    src = inspect.getsource(ap.compare)

    assert "rng.integers" in src
    assert src.count("rng.integers") == 1


def test_a_subset_scored_by_only_one_side_is_left_out():
    """Comparing a run against nothing is not a difference."""
    a = {"s1": {"V1": ("21", 1, 0.9)}, "s2": {"V1": ("21", 1, 0.9)}}
    b = {"s1": {"V1": ("21", 1, 0.1)}}

    got = ap.compare(a, b, rounds=5, seed=1)

    assert set(got) <= {"s1"}


def test_a_zero_difference_carries_no_evidence_either_way():
    """Wilcoxon's own convention: a subset that splits the two sides evenly is
    dropped rather than counted as a tie toward either."""
    assert ap.signed_rank([0.0, 0.0, 0.0])["n"] == 0
    assert ap.signed_rank([0.1, -0.2, 0.0, 0.3])["n"] == 3


def test_every_difference_in_one_direction_is_the_strongest_the_test_can_say():
    got = ap.signed_rank([0.1, 0.2, 0.3, 0.4, 0.5])

    assert got["n_positive"] == 5
    assert got["w"] == 15.0            # 1+2+3+4+5, all positive
    assert got["p_two_sided"] < 0.10


def test_the_sign_of_the_difference_follows_the_order_of_the_arguments():
    """A is first, so a positive difference means A scored higher. Getting this
    backwards inverts every conclusion in the file."""
    a = {"s1": {f"V{i}": ("21", i % 2, 0.9 if i % 2 else 0.1) for i in range(20)}}
    b = {"s1": {f"V{i}": ("21", i % 2, 0.1 if i % 2 else 0.9) for i in range(20)}}

    got = ap.compare(a, b, rounds=20, seed=1)

    assert got["s1"]["21"]["diff"] > 0


def test_a_fold_with_too_few_variants_is_skipped_rather_than_estimated():
    a = {"s1": {"V1": ("21", 1, 0.9), "V2": ("21", 0, 0.1)}}
    b = {"s1": {"V1": ("21", 1, 0.1), "V2": ("21", 0, 0.9)}}

    assert ap.compare(a, b, rounds=5, seed=1) == {}


def test_predictions_are_read_per_run_directory(tmp_path):
    run = tmp_path / "global_random_seed1"
    run.mkdir()
    (run / "predictions.jsonl").write_text(
        json.dumps({"vcv_id": "V1", "chrom": "21", "label_pathogenic": 1, "score": 0.5}) + "\n")

    got = ap.read_runs(str(tmp_path))

    assert got == {"global_random_seed1": {"V1": ("21", 1, 0.5)}}
