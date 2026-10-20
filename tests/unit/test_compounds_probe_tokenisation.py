"""The probe tokenises SMILES the way the models were trained, and pads nothing.

build_encoder's own encode() splits with list(s). That is right for a character-level
family and wrong for compounds: 579 of the 612 vocabulary entries are multi-character
SMILES tokens. Feeding a model a tokenisation it never saw is the same class of error
as dropping position_ids, which inflated a reported [MASK] loss about six-fold.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VOCAB = ROOT / "assets" / "molecules" / "vocab.txt"


def _probe():
    spec = importlib.util.spec_from_file_location(
        "compounds_downstream_eval", ROOT / "scripts" / "compounds_downstream_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(not VOCAB.exists(), reason="vocabulary not in this checkout")

# Measured: the lexer gives 23 tokens, splitting by character gives 30.
ASPIRIN_SALT = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"


def test_multi_character_tokens_survive():
    mod = _probe()
    out, skipped = mod.tokenize_smiles([ASPIRIN_SALT], str(VOCAB), 1024, "bert")
    ids, lo, hi = out[0]
    assert skipped == 0
    # [CLS] + 23 + [SEP]
    assert hi - lo == 23, hi - lo
    assert len(ids) == 25
    assert len(ids) != 32, "character splitting would give 30 + 2"


def test_br_and_cl_are_one_token_each():
    mod = _probe()
    out, _ = mod.tokenize_smiles(["Brc1ccccc1Cl"], str(VOCAB), 1024, "bert")
    ids, lo, hi = out[0]
    assert hi - lo == 10, "Br and Cl must not become B+r and C+l"


def test_gpt2_gets_no_special_tokens():
    mod = _probe()
    out, _ = mod.tokenize_smiles([ASPIRIN_SALT], str(VOCAB), 1024, "gpt2")
    ids, lo, hi = out[0]
    assert (lo, hi) == (0, 23)
    assert len(ids) == 23


def test_the_content_slice_excludes_cls_and_sep():
    """Pooling must not average a special token in."""
    mod = _probe()
    from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer

    vocab = CompoundsTokenizer(str(VOCAB), 1024).vocab
    out, _ = mod.tokenize_smiles(["CCO"], str(VOCAB), 1024, "bert")
    ids, lo, hi = out[0]
    assert ids[0] == vocab["[CLS]"] and ids[-1] == vocab["[SEP]"]
    assert ids[lo:hi] == [vocab["C"], vocab["C"], vocab["O"]]


def test_a_sequence_over_the_window_is_dropped_not_truncated():
    """Truncating would score a different molecule; the row is left out instead."""
    mod = _probe()
    out, skipped = mod.tokenize_smiles(["C" * 40], str(VOCAB), 16, "bert")
    assert out == [None]
    assert skipped == 1


def test_every_batch_is_one_length_so_nothing_is_padded():
    """_representations pads nothing and builds no mask, so batches must be uniform."""
    mod = _probe()
    smiles = ["CCO", "CCCO", "CC", "CCCCO", "CCO"]
    out, _ = mod.tokenize_smiles(smiles, str(VOCAB), 1024, "bert")
    by_len = {}
    for i, e in enumerate(out):
        by_len.setdefault(len(e[0]), []).append(i)
    for length, idxs in by_len.items():
        assert len({len(out[i][0]) for i in idxs}) == 1, length
    assert sorted(by_len) == [4, 5, 6, 7]  # CC, CCO x2, CCCO, CCCCO, each + CLS + SEP
