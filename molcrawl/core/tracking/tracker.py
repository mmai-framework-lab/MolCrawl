"""
Experiment Tracker - Main Interface
Simple API to manage experiments from each script
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import ExperimentDatabase
from .models import (
    DatasetType,
    Experiment,
    ExperimentLog,
    ExperimentStatus,
    ExperimentStep,
    ExperimentType,
    ModelType,
)


class ExperimentTracker:
    """
    experiment tracker

    Usage:
        tracker = ExperimentTracker()
        exp_id = tracker.start_experiment(
            name="GPT2 ProteinGym Training",
            experiment_type=ExperimentType.TRAINING,
            model_type=ModelType.GPT2,
            dataset_type=DatasetType.PROTEINGYM
        )

        step_id = tracker.start_step(exp_id, "data_loading", "Load dataset")
        # ... process ...
        tracker.complete_step(exp_id, step_id)

        tracker.complete_experiment(exp_id, results={"accuracy": 0.95})
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Database file path. If not specified, get from environment variable
        """
        if db_path is None:
            # defaultdatabase path of
            project_root = Path(__file__).parent.parent.parent
            db_dir = project_root / "experiment_data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "experiments.db")

        self.db = ExperimentDatabase(db_path)
        self.current_experiment_id: Optional[str] = None

    def start_experiment(
        self,
        name: str,
        experiment_type: ExperimentType,
        model_type: ModelType,
        dataset_type: DatasetType,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> str:
        """
        Start experiment

        Args:
            name: Experiment name
            experiment_type: Experiment type
            model_type: Model type
            dataset_type: Dataset type
            config: configuration information
            config_path: configuration file path
            tags: tag list
            notes: notes

        Returns:
            Experiment ID
        """
        experiment_id = (
            f"{model_type.value}_{dataset_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        # Get environment information
        environment = {
            "hostname": os.environ.get("HOSTNAME", "unknown"),
            "user": os.environ.get("USER", "unknown"),
            "python_version": os.environ.get("CONDA_PYTHON_EXE", "unknown"),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "learning_source_dir": os.environ.get("LEARNING_SOURCE_DIR", "unknown"),
            "pwd": os.getcwd(),
        }

        experiment = Experiment(
            experiment_id=experiment_id,
            experiment_name=name,
            experiment_type=experiment_type,
            model_type=model_type,
            dataset_type=dataset_type,
            status=ExperimentStatus.RUNNING,
            created_at=datetime.now(),
            started_at=datetime.now(),
            config=config or {},
            config_path=config_path,
            tags=tags or [],
            notes=notes,
            environment=environment,
        )

        self.db.save_experiment(experiment)
        self.current_experiment_id = experiment_id

        self.log(experiment_id, "INFO", f"Experiment started: {name}")

        return experiment_id

    def start_step(
        self,
        experiment_id: str,
        step_id: str,
        step_name: str,
        command: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        start step

        Args:
            experiment_id: Experiment ID
            step_id: Step ID
            step_name: Step name
            command: execution command
            metadata: metadata

        Returns:
            Step ID
        """
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        step = ExperimentStep(
            step_id=step_id,
            step_name=step_name,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now(),
            command=command,
            metadata=metadata or {},
        )

        experiment.steps.append(step)
        self.db.save_experiment(experiment)

        self.log(experiment_id, "INFO", f"Step started: {step_name}")

        return step_id

    def complete_step(
        self,
        experiment_id: str,
        step_id: str,
        output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        complete the step

        Args:
            experiment_id: Experiment ID
            step_id: Step ID
            output_path: output path
            metadata: additional metadata
        """
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        for step in experiment.steps:
            if step.step_id == step_id:
                step.status = ExperimentStatus.COMPLETED
                step.end_time = datetime.now()
                if step.start_time:
                    step.duration_seconds = (step.end_time - step.start_time).total_seconds()
                step.output_path = output_path
                if metadata:
                    step.metadata.update(metadata)
                break

        self.db.save_experiment(experiment)
        self.log(experiment_id, "INFO", f"Step completed: {step_id}")

    def fail_step(self, experiment_id: str, step_id: str, error_message: str) -> None:
        """
        put a step in a failed state

        Args:
            experiment_id: Experiment ID
            step_id: Step ID
            error_message: Error message
        """
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        for step in experiment.steps:
            if step.step_id == step_id:
                step.status = ExperimentStatus.FAILED
                step.end_time = datetime.now()
                if step.start_time:
                    step.duration_seconds = (step.end_time - step.start_time).total_seconds()
                step.error_message = error_message
                break

        self.db.save_experiment(experiment)
        self.log(experiment_id, "ERROR", f"Step failed: {step_id} - {error_message}")

    def complete_experiment(
        self,
        experiment_id: str,
        results: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        results_dir: Optional[str] = None,
    ) -> None:
        """
        Complete the experiment

        Args:
            experiment_id: Experiment ID
            results: Results information
            metrics: metrics
            results_dir: results directory
        """
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now()
        if experiment.started_at:
            experiment.total_duration_seconds = (experiment.completed_at - experiment.started_at).total_seconds()

        if results:
            experiment.results = results
        if metrics:
            experiment.metrics = metrics
        if results_dir:
            experiment.results_dir = results_dir

        self.db.save_experiment(experiment)
        self.log(experiment_id, "INFO", "Experiment completed successfully")

    def fail_experiment(self, experiment_id: str, error_message: str) -> None:
        """
        put the experiment into a failed state

        Args:
            experiment_id: Experiment ID
            error_message: Error message
        """
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.status = ExperimentStatus.FAILED
        experiment.completed_at = datetime.now()
        if experiment.started_at:
            experiment.total_duration_seconds = (experiment.completed_at - experiment.started_at).total_seconds()

        self.db.save_experiment(experiment)
        self.log(experiment_id, "ERROR", f"Experiment failed: {error_message}")

    def log(self, experiment_id: str, level: str, message: str, source: Optional[str] = None) -> None:
        """
        add log

        Args:
            experiment_id: Experiment ID
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            message: message
            source: source
        """
        log = ExperimentLog(timestamp=datetime.now(), level=level, message=message, source=source)
        self.db.add_log(experiment_id, log)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get the experiment"""
        return self.db.get_experiment(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        experiment_type: Optional[ExperimentType] = None,
        model_type: Optional[ModelType] = None,
        dataset_type: Optional[DatasetType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Experiment]:
        """Get experiment list"""
        return self.db.list_experiments(
            status=status,
            experiment_type=experiment_type,
            model_type=model_type,
            dataset_type=dataset_type,
            limit=limit,
            offset=offset,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        return self.db.get_statistics()

    def export_experiment_json(self, experiment_id: str, output_path: str) -> None:
        """Export experiment in JSON format"""
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(experiment.to_json(), encoding="utf-8")
