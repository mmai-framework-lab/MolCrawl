"""
実験トラッキングのヘルパー関数
既存のスクリプトに簡単に統合できるデコレータとコンテキストマネージャー
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
    関数を実験としてトラッキングするデコレータ

    Usage:
        @track_experiment(
            name="GPT2 Training",
            experiment_type=ExperimentType.TRAINING,
            model_type=ModelType.GPT2,
            dataset_type=DatasetType.PROTEIN_SEQUENCE
        )
        def train_model(config):
            # 訓練処理
            return {"accuracy": 0.95}
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = ExperimentTracker()

            # 実験開始
            exp_id = tracker.start_experiment(
                name=name,
                experiment_type=experiment_type,
                model_type=model_type,
                dataset_type=dataset_type,
                config=config or kwargs.get("config", {}),
                config_path=config_path,
            )

            try:
                # 関数実行
                result = func(*args, **kwargs)

                # 結果を記録
                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                    other_results = {k: v for k, v in result.items() if not isinstance(v, (int, float))}
                    tracker.complete_experiment(exp_id, results=other_results, metrics=metrics)
                else:
                    tracker.complete_experiment(exp_id)

                return result

            except Exception as e:
                # エラー処理
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
    実験をコンテキストマネージャーで管理

    Usage:
        with experiment_context(
            name="Data Preparation",
            experiment_type=ExperimentType.DATA_PREPARATION,
            model_type=ModelType.GPT2,
            dataset_type=DatasetType.PROTEIN_SEQUENCE
        ) as exp:
            # 処理
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
            """ログを追加"""
            self.tracker.log(self.experiment_id, level, message, source)

        def start_step(self, step_id: str, step_name: str, command: Optional[str] = None):
            """ステップを開始"""
            return self.tracker.start_step(self.experiment_id, step_id, step_name, command)

        def complete_step(self, step_id: str, output_path: Optional[str] = None):
            """ステップを完了"""
            self.tracker.complete_step(self.experiment_id, step_id, output_path)

        def fail_step(self, step_id: str, error_message: str):
            """ステップを失敗"""
            self.tracker.fail_step(self.experiment_id, step_id, error_message)

        def add_result(self, key: str, value: Any):
            """結果を追加"""
            self.results[key] = value

        def add_metric(self, key: str, value: float):
            """メトリクスを追加"""
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
    シンプルなステップトラッキングのコンテキストマネージャー

    Usage:
        tracker = ExperimentTracker()
        exp_id = tracker.start_experiment(...)

        with simple_track(tracker, exp_id, "Data Loading"):
            # 処理
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
