"""Evaluation benchmarks organized by task.

Each task directory groups all architecture-specific evaluation
entrypoints for that benchmark (``bert_evaluation.py``,
``gpt2_evaluation.py``, ...). Architecture differences are absorbed at
the module / CLI level so the axis of organization is the benchmark,
not the model family.
"""
