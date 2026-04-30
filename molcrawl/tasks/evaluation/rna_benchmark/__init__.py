"""Migration of the legacy RNA benchmark evaluator.

The upstream script (``molcrawl.evaluation.rna.rna_benchmark_evaluation``)
reads a JSONL of tokenised scRNA-seq cells and reports MLM accuracy and
perplexity.  The new layout keeps that contract but exposes the logic
through the adapter API so decoder and encoder architectures can share
the same pipeline.
"""

from .evaluator import RNABenchmarkEvaluator

__all__ = ["RNABenchmarkEvaluator"]
