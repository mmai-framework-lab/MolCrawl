"""ChemLLMBench evaluator - one sub-task per run."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import TASK_TYPE, TASKS, load_jsonl
from .metrics import exact_match, regression_metric, smiles_pack, text_pack

logger = logging.getLogger(__name__)


class ChemLLMBenchEvaluator(BaseEvaluator):
    task_name = "chemllmbench"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        task: str,
        jsonl_path: Path,
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
        if task not in TASKS:
            raise ValueError(f"Unknown ChemLLMBench task: {task}")
        self.task = task
        self.jsonl_path = Path(jsonl_path)
        self.max_new_tokens: int = int(self.config.get("max_new_tokens", 128))
        self.temperature: float = float(self.config.get("temperature", 0.0))

    def category(self) -> str:
        return "text_alignment"

    def load_dataset(self):
        examples = load_jsonl(self.jsonl_path)
        max_examples = self.config.get("max_examples")
        if max_examples is not None:
            examples = examples[: int(max_examples)]
        return examples

    def run_predictions(self, dataset):
        adapter = self.adapter
        if not adapter.supports("generation"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot generate."
            )
        prompts = [ex.prompt for ex in dataset]
        out = adapter.generate(
            prompts=prompts,
            num_samples=1,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        sequences = [_trim(prompt, raw) for prompt, raw in zip(prompts, out.sequences)]
        return {"predictions": sequences}

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        preds = predictions["predictions"]
        refs = [ex.answer for ex in dataset]
        task_type = TASK_TYPE[self.task]
        if task_type == "exact":
            return exact_match(preds, refs)
        if task_type == "regression":
            return regression_metric(preds, refs)
        if task_type == "text":
            return text_pack(preds, refs)
        if task_type == "smiles":
            return smiles_pack(preds, refs)
        raise RuntimeError(f"Unknown ChemLLMBench task type: {task_type}")

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update({"subtask": self.task, "task_type": TASK_TYPE[self.task], "num_examples": int(len(dataset))})
        return report


def _trim(prompt: str, raw: str) -> str:
    text = str(raw)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()
