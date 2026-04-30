"""
Experiment tracking helper functions
Decorators and context managers that can be easily integrated into existing scripts
"""

import functools
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from .models import DatasetType, ExperimentType, ModelType
from .tracker import ExperimentTracker


def track_experiment(
    name: str,
    experiment_type: ExperimentType,
    model_type: ModelType,
    dataset_type: DatasetType,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
):
    """
    Decorator to track functions as experiments

    Usage:
        @track_experiment(
            name="GPT2 Training",
            experiment_type=ExperimentType.TRAINING,
            model_type=ModelType.GPT2,
            dataset_type=DatasetType.PROTEIN_SEQUENCE
        )
        def train_model(config):
            # Training process
            return {"accuracy": 0.95}
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = ExperimentTracker()

            # Start experiment
            exp_id = tracker.start_experiment(
                name=name,
                experiment_type=experiment_type,
                model_type=model_type,
                dataset_type=dataset_type,
                config=config or kwargs.get("config", {}),
                config_path=config_path,
            )

            try:
                # function execution
                result = func(*args, **kwargs)

                # record the result
                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                    other_results = {k: v for k, v in result.items() if not isinstance(v, (int, float))}
                    tracker.complete_experiment(exp_id, results=other_results, metrics=metrics)
                else:
                    tracker.complete_experiment(exp_id)

                return result

            except Exception as e:
                # Error handling
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                tracker.fail_experiment(exp_id, error_msg)
                raise

        return wrapper

    return decorator


@contextmanager
def experiment_context(
    name: str,
    experiment_type: ExperimentType,
    model_type: ModelType,
    dataset_type: DatasetType,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
):
    """
    Manage experiments with a context manager

    Usage:
        with experiment_context(
            name="Data Preparation",
            experiment_type=ExperimentType.DATA_PREPARATION,
            model_type=ModelType.GPT2,
            dataset_type=DatasetType.PROTEIN_SEQUENCE
        ) as exp:
            # process
            exp.log("INFO", "Processing started")
            exp.start_step("step1", "Load data")
            # ...
            exp.complete_step("step1")
    """
    tracker = ExperimentTracker()

    exp_id = tracker.start_experiment(
        name=name,
        experiment_type=experiment_type,
        model_type=model_type,
        dataset_type=dataset_type,
        config=config,
        config_path=config_path,
    )

    class ExperimentContext:
        def __init__(self, experiment_id: str, tracker: ExperimentTracker):
            self.experiment_id = experiment_id
            self.tracker = tracker
            self.results: dict[str, Any] = {}
            self.metrics: dict[str, float] = {}

        def log(self, level: str, message: str, source: Optional[str] = None):
            """Add log"""
            self.tracker.log(self.experiment_id, level, message, source)

        def start_step(self, step_id: str, step_name: str, command: Optional[str] = None):
            """Start step"""
            return self.tracker.start_step(self.experiment_id, step_id, step_name, command)

        def complete_step(self, step_id: str, output_path: Optional[str] = None):
            """Complete step"""
            self.tracker.complete_step(self.experiment_id, step_id, output_path)

        def fail_step(self, step_id: str, error_message: str):
            """Failed Step"""
            self.tracker.fail_step(self.experiment_id, step_id, error_message)

        def add_result(self, key: str, value: Any):
            """Add result"""
            self.results[key] = value

        def add_metric(self, key: str, value: float):
            """Add metric"""
            self.metrics[key] = value

    ctx = ExperimentContext(exp_id, tracker)

    try:
        yield ctx
        tracker.complete_experiment(exp_id, results=ctx.results, metrics=ctx.metrics)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        tracker.fail_experiment(exp_id, error_msg)
        raise


def simple_track(tracker: ExperimentTracker, exp_id: str, step_name: str):
    """
    Simple step tracking context manager

    Usage:
        tracker = ExperimentTracker()
        exp_id = tracker.start_experiment(...)

        with simple_track(tracker, exp_id, "Data Loading"):
            # process
            pass
    """
    return _SimpleStepContext(tracker, exp_id, step_name)


class _SimpleStepContext:
    def __init__(self, tracker: ExperimentTracker, exp_id: str, step_name: str):
        self.tracker = tracker
        self.exp_id = exp_id
        self.step_id = step_name.lower().replace(" ", "_")
        self.step_name = step_name

    def __enter__(self):
        self.tracker.start_step(self.exp_id, self.step_id, self.step_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.tracker.complete_step(self.exp_id, self.step_id)
        else:
            error_msg = f"{exc_type.__name__}: {str(exc_val)}"
            self.tracker.fail_step(self.exp_id, self.step_id, error_msg)
        return False
