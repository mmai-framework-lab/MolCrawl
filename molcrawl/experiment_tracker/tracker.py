"""
実験トラッカー - メインインターフェース
各スクリプトから実験を管理するための簡易API
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
    実験トラッカー

    Usage:
        tracker = ExperimentTracker()
        exp_id = tracker.start_experiment(
            name="GPT2 ProteinGym Training",
            experiment_type=ExperimentType.TRAINING,
            model_type=ModelType.GPT2,
            dataset_type=DatasetType.PROTEINGYM
        )

        step_id = tracker.start_step(exp_id, "data_loading", "Load dataset")
        # ... 処理 ...
        tracker.complete_step(exp_id, step_id)

        tracker.complete_experiment(exp_id, results={"accuracy": 0.95})
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: データベースファイルのパス。指定しない場合は環境変数から取得
        """
        if db_path is None:
            # デフォルトのデータベースパス
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
        実験を開始

        Args:
            name: 実験名
            experiment_type: 実験タイプ
            model_type: モデルタイプ
            dataset_type: データセットタイプ
            config: 設定情報
            config_path: 設定ファイルのパス
            tags: タグリスト
            notes: メモ

        Returns:
            実験ID
        """
        experiment_id = (
            f"{model_type.value}_{dataset_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        # 環境情報を取得
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
        ステップを開始

        Args:
            experiment_id: 実験ID
            step_id: ステップID
            step_name: ステップ名
            command: 実行コマンド
            metadata: メタデータ

        Returns:
            ステップID
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
        ステップを完了

        Args:
            experiment_id: 実験ID
            step_id: ステップID
            output_path: 出力パス
            metadata: 追加メタデータ
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
        ステップを失敗状態にする

        Args:
            experiment_id: 実験ID
            step_id: ステップID
            error_message: エラーメッセージ
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
        実験を完了

        Args:
            experiment_id: 実験ID
            results: 結果情報
            metrics: メトリクス
            results_dir: 結果ディレクトリ
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
        実験を失敗状態にする

        Args:
            experiment_id: 実験ID
            error_message: エラーメッセージ
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
        ログを追加

        Args:
            experiment_id: 実験ID
            level: ログレベル (INFO, WARNING, ERROR, DEBUG)
            message: メッセージ
            source: ソース
        """
        log = ExperimentLog(timestamp=datetime.now(), level=level, message=message, source=source)
        self.db.add_log(experiment_id, log)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """実験を取得"""
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
        """実験一覧を取得"""
        return self.db.list_experiments(
            status=status,
            experiment_type=experiment_type,
            model_type=model_type,
            dataset_type=dataset_type,
            limit=limit,
            offset=offset,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return self.db.get_statistics()

    def export_experiment_json(self, experiment_id: str, output_path: str) -> None:
        """実験をJSON形式でエクスポート"""
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(experiment.to_json(), encoding="utf-8")
