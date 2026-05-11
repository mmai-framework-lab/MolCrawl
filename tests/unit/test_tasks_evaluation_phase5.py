"""Unit tests for Phase 5 molecule nat-lang evaluators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from molcrawl.tasks.evaluation._base.model_adapter import (
    GenerationOutput,
    LikelihoodOutput,
    ModelAdapter,
    ModelHandle,
    register_adapter,
)


class PromptEchoAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def generate(self, prompts=None, num_samples: int = 1, **_: Any) -> GenerationOutput:
        prompts = list(prompts or [])
        # Echo the prompt back + a short suffix so the evaluator exercises
        # the prompt-trimming path.
        return GenerationOutput(sequences=[p + " ANSWER" for p in prompts])


class ConstLLAdapter(ModelAdapter):
    def load(self) -> None:
        return None

    def score_likelihood(self, inputs: Sequence[str], **_: Any) -> LikelihoodOutput:
        return LikelihoodOutput(
            log_likelihood=[-1.0 for _ in inputs],
            num_tokens=[len(s) for s in inputs],
        )


register_adapter("phase5-gen", PromptEchoAdapter)
register_adapter("phase5-ll", ConstLLAdapter)


def test_molecule_nat_lang_runs(tmp_path: Path):
    from molcrawl.tasks.evaluation.molecule_nat_lang.evaluator import MoleculeNatLangEvaluator

    csv = tmp_path / "pairs.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN"], "caption": ["ethanol", "ethylamine"]}).to_csv(csv, index=False)

    handle = ModelHandle(arch="phase5-ll", modality="molecule_nat_lang", model_path="x")
    result = MoleculeNatLangEvaluator(
        handle=handle, output_dir=tmp_path / "out", pairs_path=csv
    ).run()
    assert "perplexity" in result.metrics


def test_chebi20_generates_both_directions(tmp_path: Path):
    from molcrawl.tasks.evaluation.chebi20.evaluator import ChEBI20Evaluator

    dataset_dir = tmp_path / "chebi20"
    dataset_dir.mkdir()
    pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN"],
            "description": ["ethanol", "ethylamine"],
        }
    ).to_csv(dataset_dir / "test.csv", index=False)

    handle = ModelHandle(arch="phase5-gen", modality="molecule_nat_lang", model_path="x")
    result = ChEBI20Evaluator(
        handle=handle,
        output_dir=tmp_path / "out",
        dataset_dir=dataset_dir,
    ).run()
    # Exact match metric should be present for both directions.
    assert "mol_to_cap.exact_match" in result.metrics
    assert "cap_to_mol.exact_match" in result.metrics


def test_chemllmbench_exact_match(tmp_path: Path):
    from molcrawl.tasks.evaluation.chemllmbench.evaluator import ChemLLMBenchEvaluator

    jsonl = tmp_path / "name_conversion.jsonl"
    lines = [
        {"prompt": "what is water in SMILES?", "answer": "ANSWER"},
        {"prompt": "what is co2?", "answer": "ANSWER"},
    ]
    jsonl.write_text("\n".join(json.dumps(line) for line in lines))

    handle = ModelHandle(arch="phase5-gen", modality="molecule_nat_lang", model_path="x")
    result = ChemLLMBenchEvaluator(
        handle=handle,
        output_dir=tmp_path / "out",
        task="name_conversion",
        jsonl_path=jsonl,
    ).run()
    assert result.metrics["exact_match"] == 1.0
