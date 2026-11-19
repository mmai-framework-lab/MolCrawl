"""Masked-marginals substitution scoring (Issue #388, M2 / step 6, step 15).

These tests pin the scoring *math* deterministically, with no model weights or
GPU: a stub adapter supplies a fixed encoder and fixed masked log-prob rows, so
the arithmetic (diff positions, sum of logP(mut) - logP(wt), reference-vs-self
== 0, indel == NaN) is checked exactly. The on-cluster numeric check against a
real BERT checkpoint (self-score == 0; single-position delta equals a
hand-computed log_softmax) runs with the full-assay sweep on a compute node.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation._base.model_adapter import (
    LikelihoodOutput,
    ModelAdapter,
    ModelHandle,
    register_adapter,
)
from molcrawl.tasks.evaluation.proteingym.masked_marginals import (
    score_substitutions_masked_marginals,
)
from molcrawl.tasks.evaluation.proteingym.metrics import (
    correlation_metrics,
    optional_binary_metrics,
)

# Token ids: CLS=0, EOS=1, MASK=2 are special; residues A/C/G/T = 3/4/5/6.
_SPECIAL = {0, 1, 2}
_CHAR2ID = {"A": 3, "C": 4, "G": 5, "T": 6}
_VOCAB = 7


def _encode_seq(seq: str) -> list[int]:
    return [0] + [_CHAR2ID[c] for c in seq] + [1]


def _rows_for(ids: list[int]) -> dict[int, np.ndarray]:
    """Deterministic masked log-prob rows for the non-special positions.

    row[pos][tok] = tok - 0.1 * pos. The ``pos`` term cancels in any
    logP(mut) - logP(wt) at a fixed position, so a substitution tok_wt->tok_mut
    contributes exactly (tok_mut - tok_wt) -- easy to hand-check.
    """
    rows: dict[int, np.ndarray] = {}
    for pos, tok in enumerate(ids):
        if tok in _SPECIAL:
            continue
        rows[pos] = np.array(
            [t - 0.1 * pos for t in range(_VOCAB)], dtype=np.float32
        )
    return rows


class StubMarginalAdapter(ModelAdapter):
    """Adapter stub exposing masked_marginal_logprobs + _encode, no weights."""

    def load(self) -> None:
        return None

    # Needed so ProteinGymEvaluator's supports("likelihood") gate passes; the
    # masked-marginals path does not actually call this.
    def score_likelihood(self, inputs: Sequence[str], **_: Any) -> LikelihoodOutput:
        return LikelihoodOutput(
            log_likelihood=[0.0 for _ in inputs], num_tokens=[0 for _ in inputs]
        )

    # No position-embedding ceiling in the stub: the raw context is effective.
    def effective_context_length(self, context_length: int) -> int:
        return int(context_length)

    def _encode(self, sequence: str, max_length: Any = None) -> list[int]:
        ids = _encode_seq(str(sequence))
        if max_length is not None and len(ids) > max_length:
            ids = ids[: max_length - 1] + [ids[-1]]
        return ids

    def masked_marginal_logprobs(
        self, reference: Any, context_length: int = 512, positions: Any = None
    ):
        eff = self.effective_context_length(context_length)
        ids = self._encode(str(reference), max_length=eff)
        return ids, _rows_for(ids)


register_adapter("stub-marginal", StubMarginalAdapter)


class TruncatingStubAdapter(StubMarginalAdapter):
    """Stub with a position-embedding ceiling, like a real BERT checkpoint.

    ``effective_context_length`` clamps to ``_MAXPOS - 2`` so a reference and
    its variants both truncate to the same window. Regression guard for the
    16-assay all-NaN bug: encoding the reference with the clamp but the variant
    with the raw context used to produce a length mismatch and NaN for every
    variant of any protein longer than the window.
    """

    _MAXPOS = 6  # CLS + 4 residues + EOS fits; longer proteins truncate.

    def effective_context_length(self, context_length: int) -> int:
        return min(int(context_length), max(1, self._MAXPOS - 2))


def test_reference_scored_against_itself_is_zero():
    adapter = StubMarginalAdapter(ModelHandle(arch="stub-marginal", modality="protein_sequence", model_path="x"))
    out = score_substitutions_masked_marginals(adapter, "ACGT", ["ACGT"])
    assert out[0] == 0.0


def test_single_substitution_equals_token_delta():
    adapter = StubMarginalAdapter(ModelHandle(arch="stub-marginal", modality="protein_sequence", model_path="x"))
    # ACGT -> AGGT: position 2 (1-indexed residue) C(4) -> G(5); delta = 5 - 4 = 1.
    out = score_substitutions_masked_marginals(adapter, "ACGT", ["AGGT"])
    assert np.isclose(out[0], 1.0, atol=1e-5)


def test_multiple_substitutions_sum():
    adapter = StubMarginalAdapter(ModelHandle(arch="stub-marginal", modality="protein_sequence", model_path="x"))
    # ACGT -> GCAT: A(3)->G(5) = +2 and G(5)->A(3) = -2; sum = 0.
    out = score_substitutions_masked_marginals(adapter, "ACGT", ["GCAT"])
    assert np.isclose(out[0], 0.0, atol=1e-5)
    # ACGT -> TCGA: A(3)->T(6) = +3 and T(6)->A(3) = -3; sum = 0.
    out2 = score_substitutions_masked_marginals(adapter, "ACGT", ["TCGA"])
    assert np.isclose(out2[0], 0.0, atol=1e-5)
    # ACGT -> TGGT: A(3)->T(6) = +3 and C(4)->G(5) = +1; sum = +4.
    out3 = score_substitutions_masked_marginals(adapter, "ACGT", ["TGGT"])
    assert np.isclose(out3[0], 4.0, atol=1e-5)


def test_indel_length_change_is_nan():
    adapter = StubMarginalAdapter(ModelHandle(arch="stub-marginal", modality="protein_sequence", model_path="x"))
    out = score_substitutions_masked_marginals(adapter, "ACGT", ["ACGTA", "ACG"])
    assert np.isnan(out[0])  # insertion
    assert np.isnan(out[1])  # deletion


def test_cost_is_reference_only():
    """The reference's rows are built once regardless of how many variants."""
    calls = {"n": 0}

    class Counting(StubMarginalAdapter):
        def masked_marginal_logprobs(self, reference, context_length=512, positions=None):
            calls["n"] += 1
            return super().masked_marginal_logprobs(reference, context_length, positions)

    adapter = Counting(ModelHandle(arch="stub-marginal", modality="protein_sequence", model_path="x"))
    muts = ["AGGT", "ACGT", "TGGT", "GCAT", "TCGA"]
    score_substitutions_masked_marginals(adapter, "ACGT", muts)
    assert calls["n"] == 1  # one reference pass for all five variants


def test_long_protein_variant_matches_reference_truncation():
    """Regression: the 16-assay all-NaN bug.

    With a position-embedding ceiling, the reference truncates to the window.
    A variant must truncate identically — an in-window substitution scores its
    token delta, and an out-of-window substitution scores 0.0 (no in-window
    diff), never NaN. Before the fix the variant was encoded with the raw
    context so its length (5) never matched the truncated reference (4) and
    every variant scored NaN.
    """
    adapter = TruncatingStubAdapter(
        ModelHandle(arch="stub-marginal", modality="protein_sequence", model_path="x")
    )
    # Reference ACGT -> window keeps CLS,A,C,EOS (first 2 residues). ctx large.
    out = score_substitutions_masked_marginals(
        adapter, "ACGT", ["AGGT", "ACGA", "ACGT"], context_length=512
    )
    # AGGT: residue-2 C(4)->G(5) is IN window -> delta +1, finite not NaN.
    assert not np.isnan(out[0]) and np.isclose(out[0], 1.0, atol=1e-5)
    # ACGA: residue-4 T->A is OUT of window -> truncated away -> no diff -> 0.0.
    assert not np.isnan(out[1]) and np.isclose(out[1], 0.0, atol=1e-5)
    # ACGT: identical -> 0.0.
    assert not np.isnan(out[2]) and np.isclose(out[2], 0.0, atol=1e-5)


def test_evaluator_masked_marginals_path(tmp_path: Path):
    """End-to-end through ProteinGymEvaluator with scoring_method set."""
    from molcrawl.tasks.evaluation.proteingym.evaluator import ProteinGymEvaluator

    wt = ["ACGT"] * 12
    # Each variant is a single substitution so the score is a known token delta.
    mut = ["AGGT", "ACGT", "TGGT", "GCGT", "ACGA", "AAGT",
           "ACAT", "ACCT", "ACGC", "TCGT", "ACTT", "CCGT"]
    score = [float(i) for i in range(12)]
    csv = tmp_path / "pgym.csv"
    pd.DataFrame(
        {"wildtype_sequence": wt, "mutated_sequence": mut, "DMS_score": score}
    ).to_csv(csv, index=False)

    handle = ModelHandle(arch="stub-marginal", modality="protein_sequence", model_path="x")
    evaluator = ProteinGymEvaluator(
        handle=handle,
        output_dir=tmp_path / "out",
        proteingym_path=csv,
        config={"scoring_method": "masked_marginals", "n_examples": None},
    )
    result = evaluator.run()
    import json

    details = json.loads(Path(result.report_paths["json"]).read_text())["details"]
    assert details.get("scoring_method") == "masked_marginals"

    jl = tmp_path / "out" / "predictions.jsonl"
    recs = [json.loads(line) for line in jl.read_text().splitlines() if line.strip()]
    assert all("score" in r for r in recs)
    # The reference-identical row (ACGT->ACGT) must score exactly 0.
    zero = [r for r in recs if r["mutated_sequence"] == "ACGT"]
    assert zero and np.isclose(zero[0]["score"], 0.0, atol=1e-5)


def test_optional_binary_metrics_drops_nan_scores():
    """A stray NaN score must not crash AUROC/AUPRC for the whole assay."""
    y_bin = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, np.nan, 0.8, 0.2, 0.7])
    out = optional_binary_metrics(y_bin, y_score)
    assert "auroc" in out and "auprc" in out
    assert np.isfinite(out["auroc"]) and np.isfinite(out["auprc"])


def test_correlation_metrics_robust_to_nan_and_degenerate():
    """Non-finite rows are dropped; an all-constant score yields NaN, not a crash."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_score = np.array([0.5, np.nan, 1.5, 2.5, 3.5])
    out = correlation_metrics(y_true, y_score)
    assert np.isfinite(out["spearman"])  # 4 finite rows, monotone -> ~1.0
    # All-constant score (every variant out of window -> 0.0): spearman is NaN
    # but the call must not raise.
    const = correlation_metrics(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    assert np.isnan(const["spearman"])
