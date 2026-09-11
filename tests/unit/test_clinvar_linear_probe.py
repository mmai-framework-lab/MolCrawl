"""How the probe splits, and what its model-free control knows.

Splits rotate over chromosomes, never over variants. Variants of one gene sit
within a few hundred bases of each other and the windows are 1,024 wide, so a
variant-level split puts overlapping sequence on both sides -- 67% of held-out
variants had a training variant within 128 bases on the old build. Rotating
chromosomes cannot overlap at all.

chrY carries 29 variants. It is never a test set; it stays on the training side
of all three folds.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "clinvar_linear_probe.py"
_spec = importlib.util.spec_from_file_location("_clinvar_probe", _SRC)
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)


def _rows(spec):
    """spec: list of (chrom, label, ref, alt, sequence)."""
    return [{"_chrom": c, "_y": y, "ref": r, "alt": a, "reference_sequence": s}
            for c, y, r, a, s in spec]


def test_the_three_folds_hold_out_21_22_and_x():
    assert lp.FOLDS == [("21",), ("22",), ("X",)]


def test_chr_y_is_never_a_test_set():
    """29 variants cannot measure anything; it belongs on the training side."""
    held = {c for fold in lp.FOLDS for c in fold}

    assert "Y" not in held
    assert lp.ALWAYS_TRAIN == ("Y",)


def test_every_variant_is_held_out_exactly_once_and_y_never():
    chroms = ["21", "22", "X", "Y"]
    counts = {c: sum(1 for fold in lp.FOLDS if c in fold) for c in chroms}

    assert counts == {"21": 1, "22": 1, "X": 1, "Y": 0}


def test_a_fold_never_trains_on_the_chromosome_it_scores():
    rows = _rows([(c, i % 2, "C", "T", "ACGT" * 8)
                  for c in ("21", "22", "X", "Y") for i in range(20)])
    feats = lp.model_free_features(rows)
    chrom = np.array([r["_chrom"] for r in rows])

    per_fold, _overall, _held = lp.probe(feats, rows, C=1.0, seed=0)

    for fold in lp.FOLDS:
        name = "chr" + fold[0]
        assert per_fold[name]["n_test"] == int(np.isin(chrom, fold).sum())
        assert per_fold[name]["n_train"] == len(rows) - per_fold[name]["n_test"]


def test_the_model_free_features_are_only_what_a_lookup_table_knows():
    """12 substitutions, CpG, GC content -- no model produces any of them."""
    rows = _rows([("21", 1, "C", "T", "GC" * 8 + "AT" * 8)])

    f = lp.model_free_features(rows)

    assert f.shape == (1, 14)                # 12 substitutions + CpG + GC
    assert f[0, :12].sum() == 1.0            # exactly one substitution type
    assert f[0, 13] == pytest.approx(0.5)    # half the window is G or C


def test_cpg_is_detected_on_either_side_of_the_site():
    """A CpG can be formed by the base before or the base after."""
    n = 16
    after = _rows([("21", 1, "C", "T", "A" * n + "CG" + "A" * (n - 2))])
    before = _rows([("21", 1, "G", "A", "A" * (n - 1) + "CG" + "A" * (n - 1))])

    assert lp.model_free_features(after)[0, 12] == 1.0
    assert lp.model_free_features(before)[0, 12] == 1.0


def test_a_non_cpg_site_is_not_flagged():
    n = 16
    rows = _rows([("21", 1, "A", "T", "A" * (2 * n))])

    assert lp.model_free_features(rows)[0, 12] == 0.0


# ---------------------------------------------------------------------------
# chrY and the per-variant record
#
# chrY is never a test fold, so its held-out scores stay NaN. np.argsort sorts NaN
# last -- it would rank those variants as the most pathogenic -- so the overall
# value must be taken over the variants that were actually scored. And the
# per-variant file is what every paired comparison is built from, so it has to
# hold exactly the held-out variants.
# ---------------------------------------------------------------------------


def _mixed_rows():
    rng = np.random.default_rng(5)
    out = []
    for c in ("21", "22", "X", "Y"):
        for i in range(40):
            y = int(rng.integers(0, 2))
            ref, alt = ("C", "T") if y else ("A", "G")
            out.append({"_chrom": c, "_y": y, "ref": ref, "alt": alt,
                        "reference_sequence": "ACGT" * 8, "vcv_id": f"V{c}{i}"})
    return out


def test_chr_y_is_left_unscored():
    rows = _mixed_rows()
    _pf, _ov, held = lp.probe(lp.model_free_features(rows), rows, C=1.0, seed=0)
    chrom = np.array([r["_chrom"] for r in rows])

    assert np.isnan(held[chrom == "Y"]).all()
    assert not np.isnan(held[chrom != "Y"]).any()


def test_the_overall_value_ignores_the_unscored_chr_y():
    rows = _mixed_rows()
    _pf, overall, held = lp.probe(lp.model_free_features(rows), rows, C=1.0, seed=0)
    y = np.array([r["_y"] for r in rows])
    ok = ~np.isnan(held)

    assert not np.isnan(overall)
    assert overall == pytest.approx(lp.auroc(y[ok], held[ok]))


def test_predictions_hold_exactly_the_held_out_variants(tmp_path):
    import json
    rows = _mixed_rows()
    _pf, _ov, held = lp.probe(lp.model_free_features(rows), rows, C=1.0, seed=0)
    path = tmp_path / "m" / "predictions.jsonl"

    n = lp.write_predictions(str(path), rows, held)
    lines = [json.loads(x) for x in path.read_text().splitlines()]

    assert n == len(lines) == 120                    # 3 folds x 40, chrY omitted
    assert {x["fold"] for x in lines} == {"chr21", "chr22", "chrX"}
    assert all(set(x) == {"vcv_id", "chrom", "fold", "label_pathogenic", "score"}
               for x in lines)
