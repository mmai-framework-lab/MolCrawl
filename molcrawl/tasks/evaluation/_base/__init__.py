"""Cross-cutting foundation for evaluation tasks.

This package provides the shared abstractions that individual evaluation
tasks in ``molcrawl.tasks.evaluation.<task>`` build on:

* :class:`~molcrawl.tasks.evaluation._base.base_evaluator.BaseEvaluator` -
  abstract evaluation loop (``loader -> predict -> metric -> report``).
* :class:`~molcrawl.tasks.evaluation._base.model_adapter.ModelAdapter` -
  uniform wrapper around GPT-2 / BERT / ChemBERTa-2 / ESM-2 / DNABERT-2 /
  RNAformer model handles.
* :class:`~molcrawl.tasks.evaluation._base.metric_registry.MetricRegistry` -
  registry of reusable metrics (perplexity, classification, regression,
  generation quality).
* :class:`~molcrawl.tasks.evaluation._base.report_writer.ReportWriter` -
  writes JSON + markdown reports and connects to
  ``molcrawl.experiment_tracker``.
"""

from .base_evaluator import BaseEvaluator, EvaluationResult
from .metric_registry import MetricRegistry, MetricSpec, default_registry
from .model_adapter import (
    ClassificationOutput,
    EmbeddingOutput,
    GenerationOutput,
    LikelihoodOutput,
    ModelAdapter,
    ModelHandle,
    RegressionOutput,
)
from .report_writer import ReportWriter

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "ModelAdapter",
    "ModelHandle",
    "ClassificationOutput",
    "RegressionOutput",
    "EmbeddingOutput",
    "GenerationOutput",
    "LikelihoodOutput",
    "MetricRegistry",
    "MetricSpec",
    "default_registry",
    "ReportWriter",
]
