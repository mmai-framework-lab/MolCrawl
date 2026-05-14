"""Foldability evaluator (structure-free proxies).

Generates protein sequences from a decoder, strips non-standard
characters, and reports distributional indicators against the
reference corpus. The 足固め upgrade adds:

- reproducible sampling (torch.manual_seed before the adapter call)
- pre-computed reference set + AA distribution (avoids re-scanning
  the ≈ 1 M-sequence reference per bootstrap iteration)
- bootstrap 95 % CIs on novelty + amino_acid_kl
- richer length-distribution diagnostics
- per-sequence predictions log (jsonl + narrative across the
  novel-long / novel-short / duplicate-of-reference quadrants)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_fasta_sequences
from .metrics import (
    AA_ALPHABET,
    amino_acid_kl,
    bootstrap_distribution_ci,
    length_stats,
    novelty_against_set,
    pfam_hit_rate,
)
from .predictions_log import write_predictions
from .splits import dedupe_generated, prepare_reference_pool

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
        self.seed: int = int(self.config.get("seed", 42))
        self.bootstrap_samples: int = int(self.config.get("bootstrap_samples", 100))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 30)
        )
        self.foldable_min_length: int = int(
            self.config.get("foldable_min_length", 50)
        )
        # When the reference FASTA is huge (RCSB pdb_seqres ≈ 1.1 M
        # sequences) we subsample for AA-distribution computation only;
        # the membership set still covers the full corpus.
        self.max_ref_for_aa: Optional[int] = self.config.get("max_ref_for_aa")

    def category(self) -> str:
        return "foldability"

    def load_dataset(self):
        sequences = load_fasta_sequences(self.reference_fasta)
        return prepare_reference_pool(
            sequences,
            max_ref_for_aa=self.max_ref_for_aa,
            seed=self.seed,
        )

    def run_predictions(self, dataset):
        adapter = self.adapter
        if not adapter.supports("generation"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot generate; "
                "foldability requires samples."
            )

        # Reproducibility: seed torch right before sampling.
        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:  # pragma: no cover
            logger.warning("torch unavailable; cannot enforce sampling seed")

        logger.info(
            "Sampling %d protein sequences (T=%.2f, top_k=%s, "
            "max_new_tokens=%d, seed=%d)",
            self.num_samples,
            self.temperature,
            self.top_k,
            self.max_new_tokens,
            self.seed,
        )
        gen_out = adapter.generate(
            num_samples=self.num_samples,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_k=self.top_k,
        )
        raw_generated = list(gen_out.sequences)
        cleaned = [_strip_non_amino_acids(s) for s in raw_generated]
        deduped = dedupe_generated(cleaned)
        return {
            "raw_generated": raw_generated,
            "cleaned_generated": cleaned,
            "generated": deduped,
            "sampling_params": gen_out.sampling_params,
        }

    def compute_metrics(self, dataset, predictions) -> Dict[str, float]:
        cleaned = predictions["cleaned_generated"]
        deduped = predictions["generated"]

        metrics: Dict[str, float] = {}
        metrics.update(length_stats(cleaned))
        metrics["novelty"] = novelty_against_set(cleaned, dataset.reference_set)
        metrics["amino_acid_kl"] = amino_acid_kl(cleaned, dataset.sequences)
        metrics["pfam_hit_rate"] = pfam_hit_rate(cleaned)

        # Bootstrap CIs (novelty + amino_acid_kl)
        bootstrap_ci = bootstrap_distribution_ci(
            cleaned,
            reference_set=dataset.reference_set,
            reference_aa_dist=dataset.aa_distribution,
            n_boot=self.bootstrap_samples,
            seed=self.seed,
        )

        # Side data for the report
        self._last_bootstrap_ci = bootstrap_ci
        self._last_dedupe_count = len(deduped)
        return metrics

    def build_report(self, metrics, dataset, predictions):
        report = super().build_report(metrics, dataset, predictions)

        bootstrap_payload: Dict[str, Dict[str, float]] = {}
        for key, (lo, hi) in (getattr(self, "_last_bootstrap_ci", {}) or {}).items():
            bootstrap_payload[key] = {"ci_lo": float(lo), "ci_hi": float(hi)}

        prediction_paths = write_predictions(
            output_dir=self.output_dir,
            raw_generated=predictions["raw_generated"],
            cleaned_generated=predictions["cleaned_generated"],
            reference_set=dataset.reference_set,
            sampling_params=predictions["sampling_params"],
            arch=self.handle.arch,
            preview_count=self.predictions_preview_count,
            foldable_min_length=self.foldable_min_length,
        )

        report.update(
            {
                "num_generated": len(predictions["cleaned_generated"]),
                "num_unique_after_dedup": getattr(
                    self, "_last_dedupe_count", None
                ),
                "num_reference": len(dataset.sequences),
                "sampling_params": predictions["sampling_params"],
                "seed": self.seed,
                "bootstrap_ci_95": bootstrap_payload,
                "reference_aa_distribution": dataset.aa_distribution,
                "artefacts": prediction_paths,
                "notes": (
                    "Structure-free foldability proxies. mean_length / "
                    "amino_acid_kl quantify whether generated sequences match "
                    "the reference's length and composition shape. novelty "
                    "checks that the model is not regurgitating the reference. "
                    "pfam_hit_rate is NaN unless HMMER + a Pfam HMM library "
                    "are wired in. The predictions narrative samples "
                    "novel-long / novel-short / duplicate-of-reference rows "
                    "so the actual generated sequences can be inspected."
                ),
            }
        )
        return report


# Built-in protein tokenizer (EsmSequenceTokenizer / BertProteinSequenceTokenizer)
# emits tokens in the form ``<cls>``, ``<eos>``, ``<pad>``, ``<unk>``,
# ``<mask>``, plus the chain-break delimiter ``|``. Decoded strings come
# back as ``"<cls> M V H L ... <eos> <pad>"`` — naively filtering
# AA-alphabet chars would keep the C / L / S inside ``<cls>`` etc.
# We strip the angle-bracketed tokens whole, then drop the
# chain-break delimiter, then keep only the standard 20 AAs.
_PROTEIN_SPECIAL_RE = re.compile(r"<[^>]*>")


def _strip_non_amino_acids(seq: str) -> str:
    """Drop characters outside the standard 20 amino-acid alphabet."""
    if not isinstance(seq, str):
        return ""
    cleaned = _PROTEIN_SPECIAL_RE.sub("", seq)
    cleaned = cleaned.replace("|", "")  # chain-break token
    cleaned = cleaned.upper()
    return "".join(ch for ch in cleaned if ch in AA_ALPHABET)
