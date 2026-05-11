"""ChEBI-20 bidirectional generation evaluator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .metrics import smiles_metrics, text_metrics
from .splits import load_all_splits

logger = logging.getLogger(__name__)


class ChEBI20Evaluator(BaseEvaluator):
    task_name = "chebi20"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        dataset_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        tracker: Optional[Any] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            handle=handle,
            output_dir=output_dir,
            config=config,
            tracker=tracker,
            experiment_id=experiment_id,
        )
        self.dataset_dir = Path(dataset_dir)
        self.direction: str = str(self.config.get("direction", "both"))
        self.max_new_tokens: int = int(self.config.get("max_new_tokens", 128))
        self.temperature: float = float(self.config.get("temperature", 0.0))
        self.prompt_mol_to_cap: str = str(
            self.config.get("prompt_mol_to_cap", "Describe the molecule: {smiles}\nDescription:")
        )
        self.prompt_cap_to_mol: str = str(
            self.config.get("prompt_cap_to_mol", "Write SMILES for: {description}\nSMILES:")
        )

    def category(self) -> str:
        return "text_alignment"

    def load_dataset(self):
        splits = load_all_splits(self.dataset_dir)
        test = splits["test"]
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            test = test.head(int(max_examples)).reset_index(drop=True)
        return {"test": test}

    def _generate(self, prompts):
        adapter = self.adapter
        out = adapter.generate(
            prompts=prompts,
            num_samples=1,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return list(out.sequences)

    def run_predictions(self, dataset):
        adapter = self.adapter
        if not adapter.supports("generation"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot generate; ChEBI-20 requires text generation."
            )
        test_df = dataset["test"]

        results: Dict[str, Any] = {"test_df": test_df, "mol_to_cap": None, "cap_to_mol": None}

        if self.direction in ("both", "mol_to_cap"):
            prompts = [self.prompt_mol_to_cap.format(smiles=s) for s in test_df["SMILES"].astype(str)]
            outputs = self._generate(prompts)
            results["mol_to_cap"] = [_trim_output(prompt, raw) for prompt, raw in zip(prompts, outputs)]

        if self.direction in ("both", "cap_to_mol"):
            prompts = [self.prompt_cap_to_mol.format(description=d) for d in test_df["description"].astype(str)]
            outputs = self._generate(prompts)
            results["cap_to_mol"] = [_trim_output(prompt, raw) for prompt, raw in zip(prompts, outputs)]

        return results

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        test_df = predictions["test_df"]
        metrics: Dict[str, float] = {}
        if predictions["mol_to_cap"] is not None:
            preds = predictions["mol_to_cap"]
            refs = test_df["description"].astype(str).tolist()
            sub = text_metrics(preds, refs)
            metrics.update({f"mol_to_cap.{k}": float(v) for k, v in sub.items()})
        if predictions["cap_to_mol"] is not None:
            preds = predictions["cap_to_mol"]
            refs = test_df["SMILES"].astype(str).tolist()
            sub = smiles_metrics(preds, refs)
            metrics.update({f"cap_to_mol.{k}": float(v) for k, v in sub.items()})
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update({"direction": self.direction, "num_test": int(len(predictions["test_df"]))})
        return report


def _trim_output(prompt: str, raw: str) -> str:
    """Remove the prompt prefix from the generated text when present."""
    text = str(raw)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()
