"""Foldability evaluator (structure-free proxies)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_fasta_sequences
from .metrics import amino_acid_kl, length_stats, novelty_vs_reference, pfam_hit_rate
from .splits import dedupe_generated

logger = logging.getLogger(__name__)


class ProteinFoldabilityEvaluator(BaseEvaluator):
    """Sample sequences from a protein decoder and report proxies."""

    task_name = "protein_foldability"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        reference_fasta: Path,
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
        self.reference_fasta = Path(reference_fasta)
        self.num_samples: int = int(self.config.get("num_samples", 100))
        self.temperature: float = float(self.config.get("temperature", 1.0))
        self.max_new_tokens: int = int(self.config.get("max_new_tokens", 256))
        self.top_k: Optional[int] = self.config.get("top_k")

    def category(self) -> str:
        return "foldability"

    def load_dataset(self):
        return {"reference": load_fasta_sequences(self.reference_fasta)}

    def run_predictions(self, dataset):
        adapter = self.adapter
        if not adapter.supports("generation"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot generate; foldability requires samples."
            )
        gen_out = adapter.generate(
            num_samples=self.num_samples,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_k=self.top_k,
        )
        sequences = [_strip_non_amino_acids(s) for s in gen_out.sequences]
        return {
            "generated": dedupe_generated(sequences),
            "raw_generated": list(gen_out.sequences),
            "sampling_params": gen_out.sampling_params,
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        generated = predictions["generated"]
        reference = dataset["reference"]
        metrics: Dict[str, float] = {}
        metrics.update(length_stats(generated))
        metrics["amino_acid_kl"] = amino_acid_kl(generated, reference)
        metrics["novelty"] = novelty_vs_reference(generated, reference)
        metrics["pfam_hit_rate"] = pfam_hit_rate(generated)
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)
        report.update(
            {
                "num_generated": len(predictions["generated"]),
                "num_reference": len(dataset["reference"]),
                "sampling_params": predictions["sampling_params"],
            }
        )
        return report


def _strip_non_amino_acids(seq: str) -> str:
    """Drop characters outside the standard 20 amino-acid alphabet."""
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(ch for ch in seq.upper() if ch in alphabet)
