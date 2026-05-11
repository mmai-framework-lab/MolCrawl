"""Evaluation benchmarks organized by task.

Each concrete task lives in ``molcrawl/tasks/evaluation/<task_name>/`` and
follows the standard layout described in
``docs/_tmp/20260422-evaluator-implementation-plan.md``::

    <task_name>/
        __init__.py
        data_preparation.py
        splits.py
        metrics.py
        evaluator.py
        visualization.py
        configs/
        README.md

The cross-cutting abstractions (base evaluator, model adapters, metric
registry, report writer) live under ``molcrawl.tasks.evaluation._base``.
Architecture differences (GPT-2 / BERT / ESM-2 / DNABERT-2 / RNAformer /
ChemBERTa-2) are absorbed at the model-adapter / CLI level rather than
by forking each evaluator per architecture.
"""
