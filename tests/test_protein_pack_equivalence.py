"""Equivalence test for the protein streaming packer.

Verifies that `pack_split_stream` (the O(context_length)-memory carry-over packer
used for full UniRef50) is byte-identical to the legacy single-global
concatenate + chunk path (the old `batch_size=-1` behaviour), on a small sample
where the legacy all-in-memory path is cheap to run.
"""
from datasets import Dataset

from molcrawl.data.protein_sequence.dataset.prepare_gpt2 import (
    concatenate_texts,
    create_chunks,
    pack_split_stream,
)


def _legacy_pack(rows, eos_token_id, context_length):
    # batch_size=-1: the whole split is one map batch -> one concat -> one chunk.
    concat = concatenate_texts({"input_ids": rows}, eos_token_id)
    return create_chunks(concat, context_length)["input_ids"]


def _stream_pack(rows, eos_token_id, context_length):
    ds = Dataset.from_dict({"input_ids": rows})
    return [r["input_ids"] for r in pack_split_stream(ds, eos_token_id, context_length)()]


def test_pack_equivalence_small():
    eos, ctx = 2, 8
    rng_rows = [
        [1, 1, 1], [4], [5, 6, 7, 8, 9, 10, 11], [3, 3], [], [9] * 15, [7, 7, 7, 7],
        [1], [2, 2, 2, 2, 2, 2], [8, 8, 8, 8, 8, 8, 8, 8, 8],
    ]
    legacy = _legacy_pack(rng_rows, eos, ctx)
    stream = _stream_pack(rng_rows, eos, ctx)
    assert stream == legacy
    # every emitted block is exactly context_length long
    assert all(len(b) == ctx for b in stream)


def test_pack_equivalence_varied_lengths():
    eos, ctx = 3, 1024
    # deterministic pseudo-random variable-length sequences
    rows = []
    v = 12345
    for _ in range(2000):
        v = (v * 1103515245 + 12345) & 0x7FFFFFFF
        length = 50 + (v % 500)
        rows.append([(v >> (i % 16)) & 0x1F for i in range(length)])
    assert _stream_pack(rows, eos, ctx) == _legacy_pack(rows, eos, ctx)
