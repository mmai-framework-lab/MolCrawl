"""ChemLLMBench evaluator - one sub-task per run.

足固め upgrade adds:

- reproducible random subsample (replaces ``examples[:max_examples]``)
- per-prompt predictions log (jsonl + correct/wrong narrative TXT)
- ``num_examples`` and ``seed`` surfaced in the report
- (NB: bootstrap CIs are NOT applied here because three of the four
  metric types — exact match on small N, BLEU/ROUGE — already collapse
  to either 0 or 1 on small samples; bootstrap on N=20 produces
  uninterpretable CIs.)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import TASK_TYPE, TASKS, ChemLLMBenchExample, load_jsonl
from .metrics import exact_match, regression_metric, smiles_pack, text_pack
from .predictions_log import write_predictions

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
        self.max_examples: Optional[int] = self.config.get("max_examples")
        self.seed: int = int(self.config.get("seed", 42))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 20)
        )

    def category(self) -> str:
        return "text_alignment"

    def load_dataset(self) -> List[ChemLLMBenchExample]:
        examples = load_jsonl(self.jsonl_path)
        if self.max_examples is not None and self.max_examples < len(examples):
            rng = random.Random(self.seed)
            indices = sorted(rng.sample(range(len(examples)), int(self.max_examples)))
            examples = [examples[i] for i in indices]
            logger.info(
                "Random subsample to %d examples (seed=%d)",
                len(examples),
                self.seed,
            )
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
        artefacts = write_predictions(
            output_dir=self.output_dir,
            examples=dataset,
            predictions=predictions["predictions"],
            task=self.task,
            task_type=TASK_TYPE[self.task],
            arch=self.handle.arch,
            preview_count=self.predictions_preview_count,
        )
        report.update(
            {
                "subtask": self.task,
                "task_type": TASK_TYPE[self.task],
                "num_examples": int(len(dataset)),
                "seed": self.seed,
                "artefacts": artefacts,
            }
        )
        return report


def _trim(prompt: str, raw: str) -> str:
    text = str(raw)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()
