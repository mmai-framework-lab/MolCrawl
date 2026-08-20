"""The pre-packing permutation must change block *contents* and nothing else.

Packing the compounds parquet in source order puts ~26 mutually similar molecules
in every block and keeps them there for the whole run, because the parquet is not
in random order. Permuting the split before packing fixes that, but it must not
move a molecule between splits, drop tokens, or change how many blocks come out.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_phase1_2_training_ready_v4_packed.py"
SPLIT_CODE = 0  # train; _split_order offsets its seed by this


def _import_packer(root, pack_order=None):
    """Import the script with the env vars it reads at module scope.

    PACK_ORDER and the output paths are read at import time, so an arm that needs a
    different value needs its own import rather than a later assignment.
    """
    env = {"LEARNING_SOURCE_DIR": str(root), "ORGANIX13_DIR": str(root)}
    if pack_order is not None:
        env["PACK_ORDER"] = pack_order
    saved = {k: os.environ.get(k) for k in list(env) + ["PACK_ORDER"]}
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location("v4_packed", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def packer(tmp_path_factory):
    return _import_packer(tmp_path_factory.mktemp("organix13"))


def _corpus(n_mol=500, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    raw_lens = rng.integers(5, 90, size=n_mol).astype(np.int64)
    row_start = np.concatenate([[0], np.cumsum(raw_lens)[:-1]]).astype(np.int64)
    row_end = np.cumsum(raw_lens).astype(np.int64)
    # Token id = molecule index + 100, so a block reveals which molecules it holds.
    all_ids_flat = np.concatenate([np.full(int(rl), 100 + i, dtype=np.int16) for i, rl in enumerate(raw_lens)])
    return raw_lens, row_start, row_end, all_ids_flat


def test_permutation_preserves_stream_length_and_block_count(packer):
    raw_lens, row_start, row_end, flat = _corpus()
    idx = np.arange(len(raw_lens))
    perm = packer._split_order(idx, SPLIT_CODE)

    a_blocks, a_len, a_rem = packer._pack_split(idx, raw_lens, row_start, row_end, flat)
    b_blocks, b_len, b_rem = packer._pack_split(perm, raw_lens, row_start, row_end, flat)

    assert (a_len, a_rem) == (b_len, b_rem)
    assert a_blocks.shape == b_blocks.shape
    assert a_blocks.shape[1] == packer.CONTEXT_LENGTH


def test_permutation_only_changes_which_tokens_land_in_the_dropped_tail(packer):
    """The two orders keep the same tokens apart from the trailing remainder.

    The stream is cut to a multiple of CONTEXT_LENGTH and the tail is dropped, and
    *which* molecules sit in that tail depends on the order — so the retained token
    multisets are not identical. They can differ by at most `remainder` tokens on
    each side, which is what bounds the difference to the drop and nothing else.
    """
    from collections import Counter

    raw_lens, row_start, row_end, flat = _corpus()
    idx = np.arange(len(raw_lens))
    perm = packer._split_order(idx, SPLIT_CODE)

    a_blocks, _, remainder = packer._pack_split(idx, raw_lens, row_start, row_end, flat)
    b_blocks, _, remainder_b = packer._pack_split(perm, raw_lens, row_start, row_end, flat)
    assert remainder == remainder_b

    a, b = Counter(a_blocks.ravel().tolist()), Counter(b_blocks.ravel().tolist())
    assert sum((a - b).values()) <= remainder
    assert sum((b - a).values()) <= remainder
    assert abs(int((a_blocks == packer.SEP_ID).sum()) - int((b_blocks == packer.SEP_ID).sum())) <= remainder


def test_same_order_is_byte_identical(packer):
    """PACK_ORDER=source has to reproduce the earlier build exactly."""
    raw_lens, row_start, row_end, flat = _corpus()
    idx = np.arange(len(raw_lens))
    a_blocks, _, _ = packer._pack_split(idx, raw_lens, row_start, row_end, flat)
    b_blocks, _, _ = packer._pack_split(idx, raw_lens, row_start, row_end, flat)
    assert np.array_equal(a_blocks, b_blocks)


def test_source_order_blocks_hold_consecutive_molecules(packer):
    """The defect the permutation removes: a block is a run of adjacent rows."""
    raw_lens, row_start, row_end, flat = _corpus()
    idx = np.arange(len(raw_lens))

    blocks, _, _ = packer._pack_split(idx, raw_lens, row_start, row_end, flat)
    first = blocks[0]
    mols = np.unique(first[first != packer.SEP_ID])
    assert np.array_equal(mols, np.arange(mols.min(), mols.max() + 1)), "source order should be contiguous"

    perm = packer._split_order(idx, SPLIT_CODE)
    shuffled_blocks, _, _ = packer._pack_split(perm, raw_lens, row_start, row_end, flat)
    first = shuffled_blocks[0]
    mols = np.unique(first[first != packer.SEP_ID])
    assert not np.array_equal(mols, np.arange(mols.min(), mols.max() + 1))


def test_each_split_gets_its_own_order(packer):
    """A shared permutation would give the three splits correlated block contents."""
    idx = np.arange(500)
    orders = [packer._split_order(idx, code) for code in (0, 1, 2)]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert not np.array_equal(orders[a], orders[b])


def test_source_order_is_returned_untouched(tmp_path):
    """PACK_ORDER=source is the documented way back to the earlier build."""
    packer = _import_packer(tmp_path, pack_order="source")
    idx = np.arange(500)
    assert np.array_equal(packer._split_order(idx, SPLIT_CODE), idx)


def test_bad_pack_order_is_rejected_at_import(tmp_path):
    """Validated at import, before 13M SMILES are tokenised."""
    with pytest.raises(SystemExit, match="PACK_ORDER"):
        _import_packer(tmp_path, pack_order="shuffle")


def test_permutation_is_reproducible(packer):
    raw_lens, row_start, row_end, flat = _corpus()
    idx = np.arange(len(raw_lens))
    assert np.array_equal(packer._split_order(idx, SPLIT_CODE), packer._split_order(idx, SPLIT_CODE))
