"""Generation-quality evaluator for compound decoders.

The MOSES protocol asks the model to free-generate ``num_samples``
SMILES and reports validity / uniqueness / novelty / internal_diversity
against a reference set. The 足固め upgrade adds:

- reproducible sampling via a torch seed before the adapter call
- bootstrap 95 % CIs over validity / uniqueness / novelty
- failure-mode classification of invalid SMILES
- length / element distribution diagnostics for both pools
- per-molecule predictions log (jsonl + narrative across the
  valid-novel / valid-seen / invalid quadrants)
- novelty cross-check against test and test_scaffolds when available
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .metrics import (
    bootstrap_distribution_ci,
    distribution_metrics,
    element_distribution,
    element_distribution_kl,
    failure_mode_summary,
    length_distribution_stats,
    optional_extended_metrics,
)
from .predictions_log import write_predictions
from .splits import ReferencePools, prepare_reference_pools

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
        self.test_limit: Optional[int] = self.config.get("test_limit")
        self.scaffolds_limit: Optional[int] = self.config.get("scaffolds_limit")
        self.include_scaffolds: bool = bool(
            self.config.get("include_scaffolds", True)
        )
        self.enable_extended: bool = bool(
            self.config.get("enable_extended_metrics", True)
        )
        self.bootstrap_samples: int = int(self.config.get("bootstrap_samples", 100))
        self.seed: int = int(self.config.get("seed", 42))
        self.predictions_preview_count: int = int(
            self.config.get("predictions_preview_count", 30)
        )

    def category(self) -> str:
        return "generation_quality"

    def load_dataset(self) -> ReferencePools:  # type: ignore[override]
        return prepare_reference_pools(
            self.reference_dir,
            train_limit=self.reference_limit,
            test_limit=self.test_limit,
            scaffolds_limit=self.scaffolds_limit,
            include_scaffolds=self.include_scaffolds,
        )

    def run_predictions(self, dataset: ReferencePools):
        adapter = self.adapter
        if not adapter.supports("generation"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} does not support generation; "
                "cannot evaluate MOSES."
            )

        # Reproducibility: seed torch before kicking generation. We rely on
        # the adapter calling torch.multinomial under the hood.
        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:  # pragma: no cover
            logger.warning("torch unavailable; cannot enforce sampling seed")

        logger.info(
            "Sampling %d molecules (T=%.2f, top_k=%s, max_new_tokens=%d, seed=%d)",
            self.num_samples,
            self.temperature,
            self.top_k,
            self.max_new_tokens,
            self.seed,
        )
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

    def compute_metrics(
        self, dataset: ReferencePools, predictions: Dict[str, Any]
    ) -> Dict[str, float]:
        generated = predictions["generated"]
        reference_train = dataset.train

        # Strip BertTokenizer special tokens + whitespace from each
        # generated string before metric computation. The raw decoded
        # form contains ``[CLS] C C ( = O ) ... [SEP] [PAD] ...`` which
        # RDKit cannot parse; downstream metric functions (validity etc.)
        # call ``Chem.MolFromSmiles`` directly so they see this cleaned
        # variant rather than the surface decode.
        cleaned = [_clean_generated_smiles(s) for s in generated]

        metrics = distribution_metrics(cleaned, reference_train)

        # Canonicalise the generated pool once. Used both for predictions
        # logging and as the pre-canonicalised input to bootstrap CIs so
        # the resampling loop does not re-canonicalise the reference set
        # (≈ 1.6 M SMILES on full MOSES train) per iteration.
        generated_canonical_list: List[Optional[str]] = _canonicalise_each(generated)
        generated_canonical_set = {c for c in generated_canonical_list if c is not None}

        bootstrap_ci = bootstrap_distribution_ci(
            cleaned,
            reference_train,
            n_boot=self.bootstrap_samples,
            seed=self.seed,
            skip_internal_diversity=True,
            generated_canonical=generated_canonical_list,
            reference_canonical_set=dataset.train_canonical,
        )

        # Cross-novelty against test / test_scaffolds
        cross_novelty: Dict[str, float] = {}
        if dataset.test_canonical:
            valid = [c for c in generated_canonical_list if c is not None]
            if valid:
                cross_novelty["novelty_vs_test"] = float(
                    sum(1 for c in valid if c not in dataset.test_canonical) / len(valid)
                )
        if dataset.scaffolds_canonical:
            valid = [c for c in generated_canonical_list if c is not None]
            if valid:
                cross_novelty["novelty_vs_scaffolds"] = float(
                    sum(
                        1 for c in valid if c not in dataset.scaffolds_canonical
                    )
                    / len(valid)
                )
        metrics.update(cross_novelty)

        # Optional reference-package extended metrics
        if self.enable_extended:
            extra = optional_extended_metrics(generated, reference_train)
            if extra:
                metrics.update(extra)

        # Stash side data for build_report
        self._last_canonicalised = generated_canonical_list
        self._last_canonical_set = generated_canonical_set
        self._last_bootstrap_ci = bootstrap_ci
        self._last_failure_modes = failure_mode_summary(
            generated, generated_canonical_list
        )
        self._last_length_stats_gen = length_distribution_stats(
            [c for c in generated_canonical_list if c is not None]
        )
        self._last_length_stats_ref = length_distribution_stats(reference_train)
        gen_elem = element_distribution(
            [c for c in generated_canonical_list if c is not None]
        )
        ref_elem = element_distribution(reference_train)
        self._last_element_dist_gen = gen_elem
        self._last_element_dist_ref = ref_elem
        self._last_element_kl = element_distribution_kl(gen_elem, ref_elem)
        return metrics

    def build_report(
        self,
        metrics: Dict[str, float],
        dataset: ReferencePools,
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        report = super().build_report(metrics, dataset, predictions)

        bootstrap_payload: Dict[str, Dict[str, float]] = {}
        for key, (lo, hi) in (getattr(self, "_last_bootstrap_ci", {}) or {}).items():
            bootstrap_payload[key] = {"ci_lo": float(lo), "ci_hi": float(hi)}

        prediction_paths = write_predictions(
            output_dir=self.output_dir,
            generated_raw=predictions["generated"],
            canonicalised=getattr(self, "_last_canonicalised", []),
            train_canonical=dataset.train_canonical,
            test_canonical=dataset.test_canonical or None,
            scaffolds_canonical=dataset.scaffolds_canonical or None,
            sampling_params=predictions["sampling_params"],
            failure_mode_counts=getattr(self, "_last_failure_modes", None),
            arch=self.handle.arch,
            preview_count=self.predictions_preview_count,
        )

        report.update(
            {
                "num_reference_train": len(dataset.train),
                "num_reference_test": len(dataset.test_canonical),
                "num_reference_scaffolds": len(dataset.scaffolds_canonical),
                "num_generated": len(predictions["generated"]),
                "sampling_params": predictions["sampling_params"],
                "seed": self.seed,
                "bootstrap_ci_95": bootstrap_payload,
                "failure_modes": getattr(self, "_last_failure_modes", None),
                "length_stats_generated_canonical": getattr(
                    self, "_last_length_stats_gen", None
                ),
                "length_stats_reference_train": getattr(
                    self, "_last_length_stats_ref", None
                ),
                "element_distribution_generated": getattr(
                    self, "_last_element_dist_gen", None
                ),
                "element_distribution_reference": getattr(
                    self, "_last_element_dist_ref", None
                ),
                "element_distribution_kl": getattr(
                    self, "_last_element_kl", None
                ),
                "artefacts": prediction_paths,
                "notes": (
                    "Score is the standard MOSES distribution-level pack. "
                    "validity / uniqueness / novelty are reported with 95 % "
                    "bootstrap CIs over the generated pool. internal_diversity "
                    "is left as a point estimate because each bootstrap "
                    "iteration is O(N^2) on Morgan fingerprints. Use the "
                    "failure_modes block plus predictions.txt to localise "
                    "where invalid molecules come from."
                ),
            }
        )
        return report


# BertTokenizer-style decoded SMILES come back as
# ``"[CLS] C C ( = O ) ... [SEP] [PAD] [PAD] ..."`` — visible special
# tokens plus single-space separators. Strip both before handing the
# string to RDKit so validity / uniqueness / novelty etc. reflect the
# model's chemical output rather than the decoder's surface form.
_SMILES_SPECIAL_TOKEN_RE = re.compile(r"\[(?:CLS|SEP|PAD|UNK|MASK)\]")
_SMILES_WHITESPACE_RE = re.compile(r"\s+")


def _clean_generated_smiles(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _SMILES_SPECIAL_TOKEN_RE.sub("", s)
    s = _SMILES_WHITESPACE_RE.sub("", s)
    return s


def _canonicalise_each(generated: Sequence[str]) -> List[Optional[str]]:
    """Return canonical SMILES per generated string (None when invalid)."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    except ImportError:
        # Without RDKit the evaluator cannot meaningfully report MOSES
        # metrics; we still return placeholders so downstream code does
        # not crash.
        return [None for _ in generated]

    out: List[Optional[str]] = []
    for s in generated:
        cleaned = _clean_generated_smiles(s)
        if not cleaned:
            out.append(None)
            continue
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            out.append(None)
            continue
        try:
            out.append(Chem.MolToSmiles(mol, canonical=True))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out
