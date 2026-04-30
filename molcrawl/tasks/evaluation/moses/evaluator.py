"""Generation-quality evaluator for compound decoders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_reference_smiles
from .metrics import distribution_metrics, optional_extended_metrics
from .splits import ensure_reference_files

logger = logging.getLogger(__name__)


class MOSESEvaluator(BaseEvaluator):
    """Evaluate SMILES generation quality on the MOSES benchmark."""

    task_name = "moses"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        reference_dir: Path,
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
        self.reference_dir = Path(reference_dir)
        self.num_samples: int = int(self.config.get("num_samples", 30000))
        self.temperature: float = float(self.config.get("temperature", 1.0))
        self.top_k: Optional[int] = self.config.get("top_k")
        self.max_new_tokens: int = int(self.config.get("max_new_tokens", 128))
        self.reference_limit: Optional[int] = self.config.get("reference_limit")
        self.enable_extended: bool = bool(self.config.get("enable_extended_metrics", True))

    def category(self) -> str:
        return "generation_quality"

    def load_dataset(self) -> Dict[str, List[str]]:
        ensure_reference_files(self.reference_dir)
        train_smiles = load_reference_smiles(
            self.reference_dir / "train.csv", limit=self.reference_limit
        )
        return {"reference": train_smiles}

    def run_predictions(self, dataset: Dict[str, List[str]]):
        adapter = self.adapter
        if not adapter.supports("generation"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} does not support generation; "
                "cannot evaluate MOSES."
            )
        logger.info("Sampling %d molecules (T=%.2f, top_k=%s)", self.num_samples, self.temperature, self.top_k)
        gen_out = adapter.generate(
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
        )
        return {
            "generated": list(gen_out.sequences),
            "sampling_params": gen_out.sampling_params,
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        generated = predictions["generated"]
        reference = dataset["reference"]
        metrics = distribution_metrics(generated, reference)
        if self.enable_extended:
            extra = optional_extended_metrics(generated, reference)
            if extra:
                metrics.update(extra)
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "num_reference": len(dataset["reference"]),
                "num_generated": len(predictions["generated"]),
                "sampling_params": predictions["sampling_params"],
            }
        )
        return report
