"""Unit tests for the Phase 1 compound evaluators.

Exercises the data-loading, split, and metric-dispatch logic that does
not need a trained model.  The evaluator classes themselves are
validated indirectly via ``Mock*Adapter`` fixtures so we can confirm
that the task pipelines wire the adapter outputs into the metric
registry correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

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


# ---------------------------------------------------------------------------
# MoleculeNet
# ---------------------------------------------------------------------------


class EmbedOnlyAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def embed(self, inputs: Sequence[str], **_: Any) -> EmbeddingOutput:
        # Represent each SMILES by (length, digit count) - enough for the
        # linear probe to distinguish tiny synthetic datasets.
        embeds = np.array(
            [[len(s), sum(ch.isdigit() for ch in s)] for s in inputs],
            dtype=float,
        )
        return EmbeddingOutput(embeddings=embeds, pooled=True)


register_adapter("embed-only", EmbedOnlyAdapter)


def _write_moleculenet_task(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCCCCCCCC", "c1ccncc1", "CCCl", "CCBr"],
            "p_np": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    task_dir = tmp_path / "bbbp"
    task_dir.mkdir()
    df.to_csv(task_dir / "raw.csv", index=False)
    (task_dir / "manifest.json").write_text("{}")
    return task_dir


def test_moleculenet_scaffold_split_shapes():
    from molcrawl.tasks.evaluation.moleculenet.splits import scaffold_split

    smiles = ["CCO"] * 10 + ["c1ccccc1"] * 10
    split = scaffold_split(smiles, val_frac=0.1, test_frac=0.1, seed=0)
    total = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    assert total == len(smiles)


def test_moleculenet_evaluator_classification(tmp_path: Path):
    from molcrawl.tasks.evaluation.moleculenet.data_preparation import get_task
    from molcrawl.tasks.evaluation.moleculenet.evaluator import MoleculeNetEvaluator

    task_dir = _write_moleculenet_task(tmp_path)
    handle = ModelHandle(arch="embed-only", modality="compounds", model_path="unused")
    evaluator = MoleculeNetEvaluator(
        handle=handle,
        output_dir=tmp_path / "out",
        task_dir=task_dir,
        task_spec=get_task("bbbp"),
        config={"split": "random", "val_frac": 0.25, "test_frac": 0.25, "seed": 0},
    )
    result = evaluator.run()
    assert result.category == "property_prediction"
    assert any(key.endswith(".accuracy") for key in result.metrics)


# ---------------------------------------------------------------------------
# MOSES
# ---------------------------------------------------------------------------


class StaticGenerator(ModelAdapter):
    def load(self) -> None:
        return None

    def generate(
        self,
        prompts=None,
        num_samples: int = 1,
        **_: Any,
    ) -> GenerationOutput:
        pool = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCCCCCCCC", "INVALID("]
        sequences = [pool[i % len(pool)] for i in range(num_samples)]
        return GenerationOutput(sequences=sequences, sampling_params={"n": num_samples})


register_adapter("static-gen", StaticGenerator)


def test_moses_evaluator_distribution_metrics(tmp_path: Path):
    from molcrawl.tasks.evaluation.moses.evaluator import MOSESEvaluator

    reference_dir = tmp_path / "moses"
    reference_dir.mkdir()
    pd.DataFrame({"SMILES": ["CCO", "CCN"]}).to_csv(reference_dir / "train.csv", index=False)
    pd.DataFrame({"SMILES": ["c1ccccc1"]}).to_csv(reference_dir / "test.csv", index=False)

    handle = ModelHandle(arch="static-gen", modality="compounds", model_path="unused")
    evaluator = MOSESEvaluator(
        handle=handle,
        output_dir=tmp_path / "out",
        reference_dir=reference_dir,
        config={
            "num_samples": 6,
            "enable_extended_metrics": False,
        },
    )
    result = evaluator.run()
    assert result.category == "generation_quality"
    metrics = result.metrics
    for key in ("validity", "uniqueness", "novelty", "internal_diversity"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


# ---------------------------------------------------------------------------
# ChEMBL scaffold held-out
# ---------------------------------------------------------------------------


class LogLikelihoodAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def score_likelihood(self, inputs: Sequence[str], **_: Any) -> LikelihoodOutput:
        # Shorter sequences get higher log-likelihood.
        ll = [-0.1 * len(s) for s in inputs]
        return LikelihoodOutput(log_likelihood=ll, num_tokens=[len(s) for s in inputs])


register_adapter("ll-only", LogLikelihoodAdapter)


def test_chembl_scaffold_heldout_perplexity(tmp_path: Path):
    from molcrawl.tasks.evaluation.chembl_scaffold_heldout.evaluator import (
        ChEMBLScaffoldHeldoutEvaluator,
    )

    heldout = tmp_path / "heldout.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCCl"]}).to_csv(heldout, index=False)

    handle = ModelHandle(arch="ll-only", modality="compounds", model_path="unused")
    evaluator = ChEMBLScaffoldHeldoutEvaluator(
        handle=handle,
        output_dir=tmp_path / "out",
        heldout_path=heldout,
    )
    result = evaluator.run()
    assert "perplexity" in result.metrics
    assert result.metrics["perplexity"] > 0.0


def test_chembl_scaffold_splits_reuses_moleculenet():
    from molcrawl.tasks.evaluation.chembl_scaffold_heldout.splits import scaffold_split
    from molcrawl.tasks.evaluation.moleculenet import splits as mol_splits

    assert scaffold_split is mol_splits.scaffold_split
