"""The two attachments §6 of the instruction of 2026-11-02 asks for.

One is the AUROC of the 21 runs over four summing ranges and three
chromosomes with a count-weighted mean; the other is the interval comparison
against the controls. Both are built from files on disk, so what is tested here
is that they refuse the inputs that would make them quietly wrong: a scoring
that predates the ranges, a run count that is not 21, and an interval file
labelled with a range it was not run on.
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_{name}", _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = _load("clinvar_zeroshot_range_table")
it = _load("clinvar_range_interval_table")


def _run(mod, argv):
    old = sys.argv
    sys.argv = ["x"] + argv
    try:
        mod.main()
    finally:
        sys.argv = old


def _tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# --- the AUROC table ------------------------------------------------------

def _aggregate(subset, per_range, step=99000):
    """One window-<subset>.json as the rescorer writes it."""
    counts = {"21": 2858, "22": 4990, "X": 13473}
    body = {}
    for rng, (n_pos, a21, a22, ax) in per_range.items():
        body[rng] = {"n_positions": n_pos, "auroc_all": ax, "n_three": 21321,
                     "auroc_three": ax}
        for c, a in (("21", a21), ("22", a22), ("X", ax)):
            body[rng][f"n_chr{c}"] = counts[c]
            body[rng][f"auroc_chr{c}"] = a
    return {"checkpoint_step": step, "ranges": body}


FOUR = {"window": (1024, 0.60, 0.70, 0.65),
        "flank128": (257, 0.62, 0.72, 0.67),
        "flank32": (65, 0.63, 0.73, 0.68),
        "point": (1, 0.58, 0.68, 0.63)}


def _scores_dir(tmp_path, n=21, ranges=None):
    d = tmp_path / "ep9-ranges"
    d.mkdir()
    for i in range(n):
        name = f"global_random_seed{i + 1}"
        json.dump(_aggregate(name, ranges if ranges is not None else FOUR),
                  open(d / f"window-{name}.json", "w"))
    return d


def test_every_run_gets_one_row_per_range(tmp_path):
    d = _scores_dir(tmp_path)
    out = tmp_path / "t.tsv"
    _run(rt, ["--scores-dir", str(d), "--out", str(out)])
    rows = _tsv(out)
    assert len(rows) == 21 * 4
    assert {r["range"] for r in rows} == {"window", "flank128", "flank32", "point"}


def test_the_weighted_mean_is_weighted_by_the_variant_counts(tmp_path):
    d = _scores_dir(tmp_path)
    out = tmp_path / "t.tsv"
    _run(rt, ["--scores-dir", str(d), "--out", str(out)])
    row = next(r for r in _tsv(out) if r["range"] == "window")
    want = (2858 * 0.60 + 4990 * 0.70 + 13473 * 0.65) / 21321
    assert float(row["auroc_weighted_mean"]) == pytest.approx(want, abs=1e-6)
    # chrX carries 63 per cent of the variants, so the mean sits nearer to it
    # than to the plain average of the three.
    assert abs(want - 0.65) < abs(want - (0.60 + 0.70 + 0.65) / 3)


def test_chr_y_is_in_no_column_of_the_table(tmp_path):
    d = _scores_dir(tmp_path)
    out = tmp_path / "t.tsv"
    _run(rt, ["--scores-dir", str(d), "--out", str(out)])
    assert not [c for c in _tsv(out)[0] if "chrY" in c or c.endswith("_Y")]
    assert int(_tsv(out)[0]["n_three"]) == 21321


def test_a_scoring_from_before_the_ranges_is_refused(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    json.dump({"auroc_chr21": 0.6, "auroc_chr22": 0.7, "auroc_chrX": 0.65},
              open(d / "window-global_random_seed1.json", "w"))
    with pytest.raises(SystemExit) as e:
        _run(rt, ["--scores-dir", str(d), "--out", str(tmp_path / "t.tsv"),
                  "--expect-runs", "1"])
    assert "ranges" in str(e.value)


def test_a_run_count_that_is_not_the_campaign_is_refused(tmp_path):
    d = _scores_dir(tmp_path, n=20)
    with pytest.raises(SystemExit) as e:
        _run(rt, ["--scores-dir", str(d), "--out", str(tmp_path / "t.tsv")])
    assert "20" in str(e.value)


# --- the interval table ---------------------------------------------------

def _interval_json(path, score_key, name="5-mer Markov", beats=(1, 1, 0)):
    json.dump({
        "baseline_key": "markov", "baseline_name": name,
        "score_key": score_key, "n_variants": 21321, "rounds": 2000,
        "seed": 1026,
        "folds": {"21": {"n": 2858, "pathogenic_rate": 0.21,
                         "baseline_auroc": 0.55},
                  "22": {"n": 4990, "pathogenic_rate": 0.20,
                         "baseline_auroc": 0.56},
                  "X": {"n": 13473, "pathogenic_rate": 0.48,
                        "baseline_auroc": 0.57}},
        "runs": {"global_random_seed1": {"per_fold": {
            c: {"auroc": 0.65, "diff_vs_baseline": 0.09,
                "diff_ci": [0.02, 0.16], "beats_baseline": bool(b)}
            for c, b in zip(("21", "22", "X"), beats)}}},
    }, open(path, "w"))
    return path


def test_one_row_per_run_range_control_and_chromosome(tmp_path):
    specs = []
    for rng in ("window", "flank128"):
        for ctl in ("5-mer Markov", "モデル不使用（14 特徴）"):
            p = tmp_path / f"i-{rng}-{len(specs)}.json"
            _interval_json(p, f"score_{rng}", name=ctl)
            specs += ["--json", f"{rng}:{p}"]
    out = tmp_path / "iv.tsv"
    _run(it, specs + ["--out", str(out)])
    rows = _tsv(out)
    assert len(rows) == 1 * 2 * 2 * 3
    assert {r["chrom"] for r in rows} == {"chr21", "chr22", "chrX"}
    assert {r["control"] for r in rows} == {"5-mer Markov", "モデル不使用（14 特徴）"}


def test_the_verdict_is_the_interval_and_not_the_point_estimate(tmp_path):
    p = _interval_json(tmp_path / "i.json", "score_window", beats=(1, 1, 0))
    out = tmp_path / "iv.tsv"
    _run(it, ["--json", f"window:{p}", "--out", str(out)])
    rows = _tsv(out)
    # chrX has the same diff as the other two and is still not a win, because
    # what is recorded is whether the interval cleared zero.
    x = next(r for r in rows if r["chrom"] == "chrX")
    assert x["diff"] == next(r for r in rows if r["chrom"] == "chr21")["diff"]
    assert x["beats_control"] == "0"
    assert sum(int(r["beats_control"]) for r in rows) == 2


def test_the_rounds_and_the_seed_travel_with_the_row(tmp_path):
    p = _interval_json(tmp_path / "i.json", "score_window")
    out = tmp_path / "iv.tsv"
    _run(it, ["--json", f"window:{p}", "--out", str(out)])
    row = _tsv(out)[0]
    assert row["rounds"] == "2000" and row["seed"] == "1026"


def test_an_interval_file_labelled_with_the_wrong_range_is_refused(tmp_path):
    p = _interval_json(tmp_path / "i.json", "score_point")
    with pytest.raises(SystemExit) as e:
        _run(it, ["--json", f"window:{p}", "--out", str(tmp_path / "iv.tsv")])
    assert "score_point" in str(e.value)
