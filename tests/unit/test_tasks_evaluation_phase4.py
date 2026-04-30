"""Unit tests for Phase 4 RNA evaluators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from molcrawl.tasks.evaluation._base.model_adapter import (
    EmbeddingOutput,
    LikelihoodOutput,
    ModelAdapter,
    ModelHandle,
    register_adapter,
)


class LengthLLAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def score_likelihood(self, inputs: Sequence[str], **_: Any) -> LikelihoodOutput:
        return LikelihoodOutput(
            log_likelihood=[-0.01 * len(s) for s in inputs],
            num_tokens=[len(s) for s in inputs],
        )


class HashEmbedAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def embed(self, inputs: Sequence[str], **_: Any) -> EmbeddingOutput:
        feats = np.array([[abs(hash(s)) % 997, len(s)] for s in inputs], dtype=float)
        return EmbeddingOutput(embeddings=feats, pooled=True)


register_adapter("phase4-ll", LengthLLAdapter)
register_adapter("phase4-embed", HashEmbedAdapter)


def test_rna_benchmark_runs(tmp_path: Path):
    from molcrawl.tasks.evaluation.rna_benchmark.evaluator import RNABenchmarkEvaluator

    jsonl = tmp_path / "rna.jsonl"
    lines = [
        {"dataset": "A", "tokens": [1, 2, 3]},
        {"dataset": "A", "tokens": [1, 2, 3, 4]},
        {"dataset": "B", "tokens": [5, 6]},
    ]
    jsonl.write_text("\n".join(json.dumps(line) for line in lines))

    handle = ModelHandle(arch="phase4-ll", modality="rna", model_path="x")
    result = RNABenchmarkEvaluator(
        handle=handle, output_dir=tmp_path / "out", rna_jsonl=jsonl
    ).run()
    assert any(k.endswith(".perplexity") for k in result.metrics)
    assert "mean.perplexity" in result.metrics


def test_tabula_sapiens_probe(tmp_path: Path):
    from molcrawl.tasks.evaluation.tabula_sapiens.evaluator import TabulaSapiensEvaluator

    jsonl = tmp_path / "tabula.jsonl"
    records = [
        {"tokens": [1, 2, 3], "cell_type": "T"},
        {"tokens": [4, 5, 6], "cell_type": "T"},
        {"tokens": [4, 5, 6, 7], "cell_type": "T"},
        {"tokens": [7, 8, 9], "cell_type": "B"},
        {"tokens": [7, 8, 9, 10], "cell_type": "B"},
        {"tokens": [10, 11], "cell_type": "B"},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records))

    handle = ModelHandle(arch="phase4-embed", modality="rna", model_path="x")
    result = TabulaSapiensEvaluator(
        handle=handle, output_dir=tmp_path / "out", jsonl_path=jsonl,
        config={"test_fraction": 0.33, "seed": 0},
    ).run()
    assert "accuracy" in result.metrics
    assert "f1_macro" in result.metrics


def test_replogle_perturb_seq(tmp_path: Path):
    from molcrawl.tasks.evaluation.replogle_perturb_seq.evaluator import (
        ReploglePerturbSeqEvaluator,
    )

    csv = tmp_path / "replogle.csv"
    rows = []
    for i, pert in enumerate(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]):
        baseline = [1.0, 1.0, 1.0, 1.0]
        perturbed = [1.0 + 0.1 * i, 1.0, 1.0 - 0.1 * i, 1.0]
        rows.append(
            {
                "perturbation": pert,
                "baseline": json.dumps(baseline),
                "perturbed": json.dumps(perturbed),
            }
        )
    pd.DataFrame(rows).to_csv(csv, index=False)

    handle = ModelHandle(arch="phase4-embed", modality="rna", model_path="x")
    result = ReploglePerturbSeqEvaluator(
        handle=handle, output_dir=tmp_path / "out", replogle_path=csv,
        config={"test_fraction": 0.4, "seed": 0},
    ).run()
    assert "spearman_mean" in result.metrics
