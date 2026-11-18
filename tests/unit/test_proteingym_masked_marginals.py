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

    def _encode(self, sequence: str, max_length: Any = None) -> list[int]:
        return _encode_seq(str(sequence))

    def masked_marginal_logprobs(
        self, reference: Any, context_length: int = 512, positions: Any = None
    ):
        ids = _encode_seq(str(reference))
        return ids, _rows_for(ids)


register_adapter("stub-marginal", StubMarginalAdapter)


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
    recs = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    assert all("score" in r for r in recs)
    # The reference-identical row (ACGT->ACGT) must score exactly 0.
    zero = [r for r in recs if r["mutated_sequence"] == "ACGT"]
    assert zero and np.isclose(zero[0]["score"], 0.0, atol=1e-5)
