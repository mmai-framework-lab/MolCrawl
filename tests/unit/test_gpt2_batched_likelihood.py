"""GPT-2 batched likelihood scoring is identical to per-sequence (Issue #388, M2).

The adapter buckets equal-length sequences and scores each bucket in one
forward pass (ProteinGym substitution variants all share the wildtype length).
Because same-length batching needs no padding, the batched result must equal
scoring one sequence at a time -- these tests pin that invariance and the
bookkeeping (length buckets, scatter back to input order, eval-position
selection, length-0/1 edges) with a deterministic stub model, no weights.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from molcrawl.tasks.evaluation._adapters.gpt2_adapter import GPT2Adapter
from molcrawl.tasks.evaluation._base.model_adapter import ModelHandle

_VOCAB = 11


class _StubModel:
    """Returns logits that depend only on (token id, position) per cell.

    Each (b, t) cell is independent of other cells, so a batched forward and a
    single-row forward produce the same logits -- exactly the nanoGPT causal
    property the batching relies on, without loading weights.
    """

    def __call__(self, ids: torch.Tensor, targets: Any = None):
        b, length = ids.shape
        v = torch.arange(_VOCAB, dtype=torch.float32).view(1, 1, _VOCAB)
        pos = torch.arange(length, dtype=torch.float32).view(1, length, 1)
        peak = (ids.to(torch.float32).unsqueeze(-1) + pos) % _VOCAB
        logits = -0.1 * (v - peak) ** 2  # (B, L, V)
        return logits, None


class _DigitTokenizer:
    """Encode a string of digits to one int token per digit (0-9 < vocab)."""

    def encode(self, s: str, add_special_tokens: bool = False):
        return [int(c) for c in s]


def _adapter(batch: int) -> GPT2Adapter:
    ad = GPT2Adapter(ModelHandle(arch="gpt2", modality="protein_sequence",
                                 model_path="x", extras={"gpt2_ll_batch": batch}))
    ad._torch = torch
    ad.device = "cpu"
    ad.model = _StubModel()
    ad.tokenizer = _DigitTokenizer()
    ad._tokenizer_kind = "hf"
    return ad


# String inputs (encoded one int token per digit). Mixed lengths, plus the
# length-0 ("") and length-1 ("2") edges.
_INPUTS = ["345", "6789", "", "2", "555", "1234", "901"]


def test_batched_equals_unbatched():
    out1 = _adapter(1).score_likelihood(_INPUTS, context_length=64)
    outN = _adapter(100).score_likelihood(_INPUTS, context_length=64)
    assert np.allclose(out1.log_likelihood, outN.log_likelihood, atol=1e-6)
    assert list(out1.num_tokens) == list(outN.num_tokens)


def test_edges_and_order_preserved():
    out = _adapter(100).score_likelihood(_INPUTS, context_length=64)
    # length 0 -> (0.0, 0); length 1 -> (0.0, 1).
    assert out.log_likelihood[2] == 0.0 and out.num_tokens[2] == 0
    assert out.log_likelihood[3] == 0.0 and out.num_tokens[3] == 1
    # num_tokens mirrors each input length for the scored rows.
    assert out.num_tokens[0] == 3 and out.num_tokens[1] == 4 and out.num_tokens[5] == 4


def test_scatter_matches_individual_calls():
    """Each input scored alone must land at its own index when batched."""
    ad = _adapter(100)
    batched = ad.score_likelihood(_INPUTS, context_length=64)
    for i, seq in enumerate(_INPUTS):
        single = ad.score_likelihood([seq], context_length=64)
        assert np.isclose(batched.log_likelihood[i], single.log_likelihood[0], atol=1e-6)
        assert batched.num_tokens[i] == single.num_tokens[0]


def test_eval_position_indices_batched_equals_unbatched():
    pos = [1, 3]
    out1 = _adapter(1).score_likelihood(_INPUTS, context_length=64, eval_position_indices=pos)
    outN = _adapter(100).score_likelihood(_INPUTS, context_length=64, eval_position_indices=pos)
    assert np.allclose(out1.log_likelihood, outN.log_likelihood, atol=1e-6)
    assert list(out1.num_tokens) == list(outN.num_tokens)
