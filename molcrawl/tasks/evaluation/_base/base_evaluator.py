"""Abstract base class for all evaluation tasks.

Concrete tasks subclass :class:`BaseEvaluator` and implement the four
extension points:

* :meth:`load_dataset` - return the data the evaluator will score on.
* :meth:`run_predictions` - run the model adapter against the data.
* :meth:`compute_metrics` - turn predictions + labels into a dict of
  scalar metrics.
* :meth:`category` - one of the task categories defined in
  ``molcrawl.experiment_tracker`` (``variant_effect``,
  ``property_prediction``, ``generation_quality``,
  ``cell_type_annotation``, ``text_alignment``, ...).

The base class orchestrates the standard ``loader -> predict -> metric ->
report`` pipeline, wires the run into :class:`ReportWriter` and, when a
tracker is supplied, logs timings and metrics into experiment_tracker.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .model_adapter import ModelAdapter, ModelHandle, build_adapter
from .report_writer import ReportWriter

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container returned by :meth:`BaseEvaluator.run`."""

    task: str
    modality: str
    arch: str
    category: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    report_paths: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "modality": self.modality,
            "arch": self.arch,
            "category": self.category,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "report_paths": dict(self.report_paths),
        }


class BaseEvaluator(ABC):
    """Common scaffolding for evaluation tasks.

    Subclasses are expected to declare ``task_name`` as a class attribute,
    returning a short identifier that matches the task directory (for
    example ``"clinvar"`` or ``"moleculenet"``).
    """

    #: Short task identifier (directory name under ``tasks/evaluation``).
    task_name: str = "unnamed_task"

    def __init__(
        self,
        handle: ModelHandle,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        tracker: Optional[Any] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        self.handle = handle
        self.config: Dict[str, Any] = dict(config or {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker
        self.experiment_id = experiment_id
        self._adapter: Optional[ModelAdapter] = None

    # ---------------------------------------------------------------
    # Adapter handling
    # ---------------------------------------------------------------

    @property
    def adapter(self) -> ModelAdapter:
        if self._adapter is None:
            self._adapter = build_adapter(self.handle)
            self._adapter.load()
        return self._adapter

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None

    # ---------------------------------------------------------------
    # Abstract interface
    # ---------------------------------------------------------------

    @abstractmethod
    def category(self) -> str:
        """Return the high-level task category."""

    @abstractmethod
    def load_dataset(self) -> Iterable[Any]:
        """Return the dataset (list of examples, dataframe, etc.)."""

    @abstractmethod
    def run_predictions(self, dataset: Any) -> Any:
        """Produce predictions over the dataset using ``self.adapter``."""

    @abstractmethod
    def compute_metrics(self, dataset: Any, predictions: Any) -> Dict[str, float]:
        """Compute a mapping of metric name -> scalar."""

    # ---------------------------------------------------------------
    # Optional hooks
    # ---------------------------------------------------------------

    def build_report(
        self, metrics: Dict[str, float], dataset: Any, predictions: Any
    ) -> Dict[str, Any]:
        """Return the payload written to JSON and markdown reports.

        Default implementation returns just ``metrics`` and the evaluator
        config.  Tasks can override this to include confusion matrices,
        distribution summaries, etc.
        """
        return {
            "task": self.task_name,
            "modality": self.handle.modality,
            "arch": self.handle.arch,
            "model_path": self.handle.model_path,
            "category": self.category(),
            "config": self.config,
            "metrics": metrics,
        }

    # ---------------------------------------------------------------
    # Orchestration
    # ---------------------------------------------------------------

    def run(self) -> EvaluationResult:
        """Execute the full evaluation pipeline and persist the report."""
        start = time.perf_counter()
        logger.info(
            "Running evaluator task=%s arch=%s modality=%s",
            self.task_name,
            self.handle.arch,
            self.handle.modality,
        )

        step_id = f"{self.task_name}_eval"
        self._start_step(step_id)
        try:
            dataset = self.load_dataset()
            predictions = self.run_predictions(dataset)
            metrics = self.compute_metrics(dataset, predictions)
        except Exception as exc:
            self._fail_step(step_id, str(exc))
            raise
        finally:
            self.close()

        report = self.build_report(metrics, dataset, predictions)
        writer = ReportWriter(self.output_dir)
        paths = writer.write(
            task=self.task_name,
            modality=self.handle.modality,
            arch=self.handle.arch,
            category=self.category(),
            metrics=metrics,
            report=report,
        )

        self._complete_step(step_id, output_path=str(self.output_dir))
        self._log_metrics(metrics)

        duration = time.perf_counter() - start
        logger.info(
            "Completed evaluator task=%s in %.2fs -> %s",
            self.task_name,
            duration,
            self.output_dir,
        )

        return EvaluationResult(
            task=self.task_name,
            modality=self.handle.modality,
            arch=self.handle.arch,
            category=self.category(),
            metrics=metrics,
            metadata={"duration_seconds": duration, "config": self.config},
            report_paths=paths,
        )

    # ---------------------------------------------------------------
    # experiment_tracker glue (best-effort, no-ops when unavailable)
    # ---------------------------------------------------------------

    def _start_step(self, step_id: str) -> None:
        if self.tracker is None or self.experiment_id is None:
            return
        try:
            self.tracker.start_step(self.experiment_id, step_id, step_id)
        except Exception:  # pragma: no cover - tracking must not break runs
            logger.exception("Failed to start tracker step %s", step_id)

    def _complete_step(self, step_id: str, output_path: Optional[str] = None) -> None:
        if self.tracker is None or self.experiment_id is None:
            return
        try:
            self.tracker.complete_step(self.experiment_id, step_id, output_path=output_path)
        except Exception:
            logger.exception("Failed to complete tracker step %s", step_id)

    def _fail_step(self, step_id: str, error: str) -> None:
        if self.tracker is None or self.experiment_id is None:
            return
        try:
            self.tracker.fail_step(self.experiment_id, step_id, error)
        except Exception:
            logger.exception("Failed to mark tracker step %s as failed", step_id)

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        if self.tracker is None or self.experiment_id is None:
            return
        try:
            self.tracker.complete_experiment(self.experiment_id, metrics=metrics)
        except Exception:
            logger.exception("Failed to push metrics to tracker")
