"""Unit tests for Phase 3 genome evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation._base.model_adapter import (
    EmbeddingOutput,
    LikelihoodOutput,
    ModelAdapter,
    ModelHandle,
    register_adapter,
)


class LengthLikelihoodAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def score_likelihood(self, inputs: Sequence[str], **_: Any) -> LikelihoodOutput:
        return LikelihoodOutput(
            log_likelihood=[-0.05 * len(s) for s in inputs],
            num_tokens=[len(s) for s in inputs],
        )


class SimpleEmbedAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def embed(self, inputs: Sequence[str], **_: Any) -> EmbeddingOutput:
        feats = np.array(
            [[len(s), s.count("A"), s.count("C"), s.count("G"), s.count("T")] for s in inputs],
            dtype=float,
        )
        return EmbeddingOutput(embeddings=feats, pooled=True)


register_adapter("phase3-ll", LengthLikelihoodAdapter)
register_adapter("phase3-embed", SimpleEmbedAdapter)


def test_cosmic_runs(tmp_path: Path):
    from molcrawl.tasks.evaluation.cosmic.evaluator import CosmicEvaluator

    csv = tmp_path / "cosmic.csv"
    pd.DataFrame(
        {
            "reference_sequence": ["ACGT", "ACGT", "ACGT", "ACGT"],
            "variant_sequence": ["AAGT", "ACCT", "ACGA", "ACGC"],
            "FATHMM_PREDICTION": ["PATHOGENIC", "NEUTRAL", "PATHOGENIC", "NEUTRAL"],
        }
    ).to_csv(csv, index=False)
    handle = ModelHandle(arch="phase3-ll", modality="genome_sequence", model_path="x")
    result = CosmicEvaluator(
        handle=handle, output_dir=tmp_path / "out", cosmic_path=csv
    ).run()
    assert "accuracy" in result.metrics


def test_omim_runs(tmp_path: Path):
    from molcrawl.tasks.evaluation.omim.evaluator import OMIMEvaluator

    csv = tmp_path / "omim.csv"
    pd.DataFrame(
        {
            "reference_sequence": ["ACGT"] * 4,
            "variant_sequence": ["AAGT", "ACCT", "ACGA", "ACGC"],
            "disease_category": ["known disease", "unknown", "mendelian", "control"],
        }
    ).to_csv(csv, index=False)
    handle = ModelHandle(arch="phase3-ll", modality="genome_sequence", model_path="x")
    result = OMIMEvaluator(
        handle=handle, output_dir=tmp_path / "out", omim_path=csv
    ).run()
    assert result.category == "variant_effect"


def test_gnomad_runs(tmp_path: Path):
    from molcrawl.tasks.evaluation.gnomad_af_correlation.evaluator import GnomadAFEvaluator

    csv = tmp_path / "gnomad.csv"
    pd.DataFrame(
        {
            "reference_sequence": ["AAAA", "AACG", "AAGG", "AACC"],
            "variant_sequence": ["AAAG", "AACG", "AAGG", "ACCC"],
            "allele_frequency": [0.01, 0.5, 0.001, 0.3],
        }
    ).to_csv(csv, index=False)
    handle = ModelHandle(arch="phase3-ll", modality="genome_sequence", model_path="x")
    result = GnomadAFEvaluator(
        handle=handle, output_dir=tmp_path / "out", gnomad_path=csv
    ).run()
    assert "spearman" in result.metrics


def test_gue_probe(tmp_path: Path):
    from molcrawl.tasks.evaluation.gue.data_preparation import get_spec
    from molcrawl.tasks.evaluation.gue.evaluator import GUEEvaluator

    task_dir = tmp_path / "prom_300_all"
    task_dir.mkdir()
    pd.DataFrame(
        {
            "sequence": ["AAAA", "AAAG", "AAGG", "AGGG", "GGGG", "CCCC"],
            "label": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(task_dir / "train.csv", index=False)
    pd.DataFrame(
        {"sequence": ["AAGG", "GGGG"], "label": [0, 1]}
    ).to_csv(task_dir / "test.csv", index=False)

    handle = ModelHandle(arch="phase3-embed", modality="genome_sequence", model_path="x")
    result = GUEEvaluator(
        handle=handle,
        output_dir=tmp_path / "out",
        task_dir=task_dir,
        task_spec=get_spec("prom_300_all"),
    ).run()
    assert "accuracy" in result.metrics
    assert "mcc" in result.metrics
