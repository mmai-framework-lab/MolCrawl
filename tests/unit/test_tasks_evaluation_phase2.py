"""Unit tests for the Phase 2 protein evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from molcrawl.tasks.evaluation._base.model_adapter import (
    EmbeddingOutput,
    GenerationOutput,
    LikelihoodOutput,
    ModelAdapter,
    ModelHandle,
    register_adapter,
)


class DeterministicEmbedAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def embed(self, inputs: Sequence[str], **_: Any) -> EmbeddingOutput:
        feats = np.array([[len(s), s.count("A"), s.count("G")] for s in inputs], dtype=float)
        return EmbeddingOutput(embeddings=feats, pooled=True)


class StaticGeneratorAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def generate(self, prompts=None, num_samples: int = 1, **_: Any) -> GenerationOutput:
        pool = ["MKTA", "MKTG", "ACDE", "VLIF"]
        return GenerationOutput(sequences=[pool[i % len(pool)] for i in range(num_samples)])


class LengthLikelihoodAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def score_likelihood(self, inputs: Sequence[str], **_: Any) -> LikelihoodOutput:
        return LikelihoodOutput(
            log_likelihood=[-0.05 * len(s) for s in inputs],
            num_tokens=[len(s) for s in inputs],
        )


register_adapter("phase2-embed", DeterministicEmbedAdapter)
register_adapter("phase2-gen", StaticGeneratorAdapter)
register_adapter("phase2-ll", LengthLikelihoodAdapter)


def test_proteingym_spearman(tmp_path: Path):
    from molcrawl.tasks.evaluation.proteingym.evaluator import ProteinGymEvaluator

    csv = tmp_path / "pgym.csv"
    pd.DataFrame(
        {
            "wildtype_sequence": ["AAAA", "AAAG", "AAAG", "AAAG"],
            "mutated_sequence": ["AAAA", "AAAG", "AAGG", "AAAA"],
            "DMS_score": [0.0, -0.1, -0.5, -1.0],
        }
    ).to_csv(csv, index=False)
    handle = ModelHandle(arch="phase2-ll", modality="protein_sequence", model_path="x")
    evaluator = ProteinGymEvaluator(
        handle=handle, output_dir=tmp_path / "out", proteingym_path=csv
    )
    result = evaluator.run()
    assert "spearman" in result.metrics
    assert "pearson" in result.metrics


def test_deeploc_multiclass(tmp_path: Path):
    from molcrawl.tasks.evaluation.deeploc.evaluator import DeepLocEvaluator

    csv = tmp_path / "deeploc.csv"
    pd.DataFrame(
        {
            "sequence": [
                "AAAA", "AAAG", "AAGG", "AGGG", "GGGG",
                "CCCA", "CCCG", "CCGG", "CGGG", "GGGG",
            ],
            "localisation": [
                "Cytoplasm", "Cytoplasm", "Cytoplasm", "Cytoplasm", "Cytoplasm",
                "Nucleus", "Nucleus", "Nucleus", "Nucleus", "Nucleus",
            ],
        }
    ).to_csv(csv, index=False)
    handle = ModelHandle(arch="phase2-embed", modality="protein_sequence", model_path="x")
    evaluator = DeepLocEvaluator(
        handle=handle, output_dir=tmp_path / "out", deeploc_path=csv,
        config={"test_fraction": 0.3, "seed": 0},
    )
    result = evaluator.run()
    for key in ("accuracy", "f1_macro", "mcc"):
        assert key in result.metrics


def test_foldability_proxies(tmp_path: Path):
    from molcrawl.tasks.evaluation.protein_foldability.evaluator import (
        ProteinFoldabilityEvaluator,
    )

    fasta = tmp_path / "reference.fasta"
    fasta.write_text(">ref1\nMKTA\n>ref2\nACDE\n")
    handle = ModelHandle(arch="phase2-gen", modality="protein_sequence", model_path="x")
    evaluator = ProteinFoldabilityEvaluator(
        handle=handle,
        output_dir=tmp_path / "out",
        reference_fasta=fasta,
        config={"num_samples": 8, "max_new_tokens": 16},
    )
    result = evaluator.run()
    assert result.metrics["mean_length"] > 0.0
    assert 0.0 <= result.metrics["novelty"] <= 1.0
    assert "amino_acid_kl" in result.metrics


def test_tape_regression_probe(tmp_path: Path):
    from molcrawl.tasks.evaluation.tape.data_preparation import get_spec
    from molcrawl.tasks.evaluation.tape.evaluator import TAPEEvaluator

    task_dir = tmp_path / "fluorescence"
    task_dir.mkdir()
    records_train = [
        {"primary": "AAAG", "log_fluorescence": 0.1},
        {"primary": "AAGG", "log_fluorescence": 0.4},
        {"primary": "AGGG", "log_fluorescence": 0.7},
        {"primary": "GGGG", "log_fluorescence": 1.0},
    ]
    records_valid = [
        {"primary": "AAGA", "log_fluorescence": 0.2},
        {"primary": "AGGA", "log_fluorescence": 0.6},
    ]
    (task_dir / "fluorescence_train.json").write_text(
        "\n".join(__import__("json").dumps(r) for r in records_train)
    )
    (task_dir / "fluorescence_valid.json").write_text(
        "\n".join(__import__("json").dumps(r) for r in records_valid)
    )
    handle = ModelHandle(arch="phase2-embed", modality="protein_sequence", model_path="x")
    evaluator = TAPEEvaluator(
        handle=handle,
        output_dir=tmp_path / "out",
        task_dir=task_dir,
        task_spec=get_spec("fluorescence"),
    )
    result = evaluator.run()
    for key in ("rmse", "spearman", "pearson"):
        assert key in result.metrics
