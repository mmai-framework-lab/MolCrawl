"""Score likelihoods and correlate with gnomAD allele frequency.

The evaluator samples variants with AF-log-bin stratification (rare vs
common) so the Spearman / Pearson correlation is not dominated by the
heavy rare-variant tail of gnomAD, reports 95 % bootstrap CIs as
primary uncertainty quantification, and additionally emits a per-bin
correlation table so the reader can see whether the model's signal
concentrates at a particular AF regime.

Per-variant artefacts (``predictions.jsonl`` / ``predictions.txt``)
mirror the ClinVar evaluator but the narrative previews by AF rank
rather than by correct/wrong quadrants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation import _adapters  # noqa: F401 - registers adapters
from molcrawl.tasks.evaluation._base import BaseEvaluator, ModelHandle

from .data_preparation import load_gnomad
from .metrics import (
    bootstrap_correlation_ci,
    correlation_metrics,
    per_bin_correlation,
    score_distribution_stats,
)
from .predictions_log import write_predictions
from .splits import sample_gnomad

logger = logging.getLogger(__name__)

_MIN_ROWS_FOR_CORRELATION = 10


class GnomadAFEvaluator(BaseEvaluator):
    task_name = "gnomad_af_correlation"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        gnomad_path: Path,
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
        self.gnomad_path = Path(gnomad_path)
        self.context_length: int = int(self.config.get("context_length", 512))

    def category(self) -> str:
        return "variant_effect"

    def load_dataset(self) -> pd.DataFrame:
        df = load_gnomad(self.gnomad_path)

        n_per_bin = self.config.get("n_per_bin")
        max_examples = self.config.get("max_examples")
        if n_per_bin is None and max_examples is not None:
            # Backwards compatibility: redistribute the legacy cap across
            # the 6 default AF bins so old smoke scripts still draw a
            # rank-diverse sample rather than a head() slice.
            n_per_bin = max(1, int(max_examples) // 6)
            logger.warning(
                "GnomadAFEvaluator: max_examples=%s is deprecated; "
                "re-interpreting as n_per_bin=%d across 6 AF bins.",
                max_examples,
                n_per_bin,
            )

        seed = int(self.config.get("seed", 42))
        sampled = sample_gnomad(df, n_per_bin=n_per_bin, seed=seed)

        self._last_sampling = {
            "n_per_bin": n_per_bin,
            "seed": seed,
            "total_rows_in_file": int(len(df)),
        }
        return sampled

    def run_predictions(self, dataset: pd.DataFrame) -> Dict[str, np.ndarray]:
        adapter = self.adapter
        if not adapter.supports("likelihood"):
            raise RuntimeError(
                f"Adapter {type(adapter).__name__} cannot score likelihoods."
            )
        ref_out = adapter.score_likelihood(
            dataset["reference_sequence"].astype(str).tolist(),
            context_length=self.context_length,
        )
        var_out = adapter.score_likelihood(
            dataset["variant_sequence"].astype(str).tolist(),
            context_length=self.context_length,
        )
        ref_ll = np.asarray(ref_out.log_likelihood, dtype=float)
        var_ll = np.asarray(var_out.log_likelihood, dtype=float)
        # Common alleles are expected to look "natural" to the model, so we
        # score as LL(var) − LL(ref): a positive Spearman(AF, score) means
        # the model ranks more-frequent alleles higher.
        scores = var_ll - ref_ll
        return {
            "scores": scores,
            "reference_log_likelihood": ref_ll,
            "variant_log_likelihood": var_ll,
        }

    def compute_metrics(
        self, dataset: pd.DataFrame, predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        af = dataset["allele_frequency"].to_numpy(dtype=float)
        scores = predictions["scores"]
        ref_ll = predictions["reference_log_likelihood"]
        var_ll = predictions["variant_log_likelihood"]

        metrics: Dict[str, float] = {}
        if len(af) < _MIN_ROWS_FOR_CORRELATION:
            logger.warning(
                "GnomadAFEvaluator: skipping correlation metrics (%d < %d rows)",
                len(af),
                _MIN_ROWS_FOR_CORRELATION,
            )
            self._last_bootstrap = {}
            self._last_per_bin = {}
            self._last_score_distribution = {}
            return metrics

        metrics.update(correlation_metrics(af, scores))

        n_boot = int(self.config.get("bootstrap_samples", 200))
        self._last_bootstrap = bootstrap_correlation_ci(
            af, scores, n_boot=n_boot, seed=int(self.config.get("seed", 42))
        )
        self._last_per_bin = per_bin_correlation(af, scores)
        self._last_score_distribution = score_distribution_stats(
            af, ref_ll, var_ll, scores
        )
        return metrics

    def build_report(
        self,
        metrics: Dict[str, float],
        dataset: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        report = super().build_report(metrics, dataset, predictions)

        sampling = getattr(self, "_last_sampling", None)
        per_bin = getattr(self, "_last_per_bin", None)
        bootstrap_ci = getattr(self, "_last_bootstrap", None)
        score_distribution = getattr(self, "_last_score_distribution", None)

        # Format the bootstrap CIs as serialisable dicts.
        bootstrap_payload: Dict[str, Dict[str, float]] = {}
        if bootstrap_ci:
            for key, (lo, hi) in bootstrap_ci.items():
                bootstrap_payload[key] = {"ci_lo": float(lo), "ci_hi": float(hi)}

        preview_count = int(self.config.get("predictions_preview_count", 20))
        prediction_paths = write_predictions(
            output_dir=self.output_dir,
            dataset=dataset,
            predictions=predictions,
            score_distribution=score_distribution,
            sampling=sampling,
            arch=self.handle.arch,
            modality=self.handle.modality,
            preview_count=preview_count,
        )

        report.update(
            {
                "num_variants": int(len(dataset)),
                "sampling": sampling,
                "af_distribution": {
                    "min": float(dataset["allele_frequency"].min())
                    if len(dataset)
                    else None,
                    "max": float(dataset["allele_frequency"].max())
                    if len(dataset)
                    else None,
                    "median": float(dataset["allele_frequency"].median())
                    if len(dataset)
                    else None,
                },
                "per_bin_correlation": per_bin,
                "bootstrap_ci_95": bootstrap_payload,
                "score_distribution": score_distribution,
                "artefacts": prediction_paths,
                "notes": (
                    "Score is LL(var) − LL(ref): higher means the model ranks "
                    "the variant allele above the reference. Positive Spearman "
                    "with AF is the expected direction (common alleles look "
                    "'natural'). Bootstrap CI uses resampling over rows; 95 % "
                    "intervals that straddle 0 mean the correlation is not "
                    "statistically distinguishable from zero."
                ),
            }
        )
        return report
