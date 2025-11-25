"""
実験管理データベース - SQLiteベース
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from .models import (
    Experiment,
    ExperimentStatus,
    ExperimentType,
    ModelType,
    DatasetType,
    ExperimentStep,
    ExperimentLog,
)


class ExperimentDatabase:
    """実験管理データベース"""

    def __init__(self, db_path: str = "experiments.db"):
        """
        Args:
            db_path: データベースファイルのパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def get_connection(self):
        """データベース接続のコンテキストマネージャー"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _initialize_database(self):
        """データベースの初期化"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 実験テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    experiment_type TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    dataset_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    total_duration_seconds REAL,
                    config_path TEXT,
                    config TEXT,
                    results_dir TEXT,
                    results TEXT,
                    metrics TEXT,
                    tags TEXT,
                    notes TEXT,
                    environment TEXT
                )
            """)

            # ステップテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    command TEXT,
                    output_path TEXT,
                    error_message TEXT,
                    metadata TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            # ログテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            # インデックス作成
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_status
                ON experiments(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_type
                ON experiments(experiment_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_created
                ON experiments(created_at)
            """)

    def save_experiment(self, experiment: Experiment) -> None:
        """実験を保存"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, experiment_name, experiment_type,
                    model_type, dataset_type, status,
                    created_at, started_at, completed_at,
                    total_duration_seconds, config_path, config,
                    results_dir, results, metrics,
                    tags, notes, environment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment.experiment_id,
                    experiment.experiment_name,
                    experiment.experiment_type.value,
                    experiment.model_type.value,
                    experiment.dataset_type.value,
                    experiment.status.value,
                    experiment.created_at.isoformat(),
                    experiment.started_at.isoformat() if experiment.started_at else None,
                    experiment.completed_at.isoformat() if experiment.completed_at else None,
                    experiment.total_duration_seconds,
                    experiment.config_path,
                    json.dumps(experiment.config),
                    experiment.results_dir,
                    json.dumps(experiment.results),
                    json.dumps(experiment.metrics),
                    json.dumps(experiment.tags),
                    experiment.notes,
                    json.dumps(experiment.environment),
                ),
            )

            # ステップを保存
            cursor.execute(
                "DELETE FROM experiment_steps WHERE experiment_id = ?",
                (experiment.experiment_id,),
            )
            for step in experiment.steps:
                cursor.execute(
                    """
                    INSERT INTO experiment_steps (
                        experiment_id, step_id, step_name, status,
                        start_time, end_time, duration_seconds,
                        command, output_path, error_message, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        experiment.experiment_id,
                        step.step_id,
                        step.step_name,
                        step.status.value,
                        step.start_time.isoformat() if step.start_time else None,
                        step.end_time.isoformat() if step.end_time else None,
                        step.duration_seconds,
                        step.command,
                        step.output_path,
                        step.error_message,
                        json.dumps(step.metadata),
                    ),
                )

    def add_log(self, experiment_id: str, log: ExperimentLog) -> None:
        """ログを追加"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiment_logs (
                    experiment_id, timestamp, level, message, source
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    experiment_id,
                    log.timestamp.isoformat(),
                    log.level,
                    log.message,
                    log.source,
                ),
            )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """実験を取得"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 実験情報を取得
            cursor.execute(
                """
                SELECT * FROM experiments WHERE experiment_id = ?
            """,
                (experiment_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            # ステップを取得
            cursor.execute(
                """
                SELECT * FROM experiment_steps WHERE experiment_id = ?
                ORDER BY id
            """,
                (experiment_id,),
            )
            steps_rows = cursor.fetchall()

            # ログを取得
            cursor.execute(
                """
                SELECT * FROM experiment_logs WHERE experiment_id = ?
                ORDER BY timestamp
            """,
                (experiment_id,),
            )
            logs_rows = cursor.fetchall()

            # Experimentオブジェクトを構築
            experiment_data = dict(row)
            experiment_data["experiment_type"] = ExperimentType(experiment_data["experiment_type"])
            experiment_data["model_type"] = ModelType(experiment_data["model_type"])
            experiment_data["dataset_type"] = DatasetType(experiment_data["dataset_type"])
            experiment_data["status"] = ExperimentStatus(experiment_data["status"])
            experiment_data["created_at"] = datetime.fromisoformat(experiment_data["created_at"])

            if experiment_data.get("started_at"):
                experiment_data["started_at"] = datetime.fromisoformat(experiment_data["started_at"])
            if experiment_data.get("completed_at"):
                experiment_data["completed_at"] = datetime.fromisoformat(experiment_data["completed_at"])

            experiment_data["config"] = json.loads(experiment_data["config"]) if experiment_data["config"] else {}
            experiment_data["results"] = json.loads(experiment_data["results"]) if experiment_data["results"] else {}
            experiment_data["metrics"] = json.loads(experiment_data["metrics"]) if experiment_data["metrics"] else {}
            experiment_data["tags"] = json.loads(experiment_data["tags"]) if experiment_data["tags"] else []
            experiment_data["environment"] = (
                json.loads(experiment_data["environment"]) if experiment_data["environment"] else {}
            )

            # ステップを構築
            steps = []
            for step_row in steps_rows:
                step_data = dict(step_row)
                step_data["status"] = ExperimentStatus(step_data["status"])
                if step_data.get("start_time"):
                    step_data["start_time"] = datetime.fromisoformat(step_data["start_time"])
                if step_data.get("end_time"):
                    step_data["end_time"] = datetime.fromisoformat(step_data["end_time"])
                step_data["metadata"] = json.loads(step_data["metadata"]) if step_data["metadata"] else {}
                # 不要なフィールドを削除
                del step_data["id"]
                del step_data["experiment_id"]
                steps.append(ExperimentStep(**step_data))

            # ログを構築
            logs = []
            for log_row in logs_rows:
                log_data = dict(log_row)
                log_data["timestamp"] = datetime.fromisoformat(log_data["timestamp"])
                # 不要なフィールドを削除
                del log_data["id"]
                del log_data["experiment_id"]
                logs.append(ExperimentLog(**log_data))

            experiment_data["steps"] = steps
            experiment_data["logs"] = logs

            return Experiment(**experiment_data)

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
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT experiment_id FROM experiments WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status.value)
            if experiment_type:
                query += " AND experiment_type = ?"
                params.append(experiment_type.value)
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type.value)
            if dataset_type:
                query += " AND dataset_type = ?"
                params.append(dataset_type.value)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            experiments = []
            for row in rows:
                exp = self.get_experiment(row["experiment_id"])
                if exp:
                    experiments.append(exp)

            return experiments

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # 全実験数
            cursor.execute("SELECT COUNT(*) as count FROM experiments")
            stats["total_experiments"] = cursor.fetchone()["count"]

            # ステータス別
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM experiments
                GROUP BY status
            """)
            stats["by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}

            # タイプ別
            cursor.execute("""
                SELECT experiment_type, COUNT(*) as count
                FROM experiments
                GROUP BY experiment_type
            """)
            stats["by_type"] = {row["experiment_type"]: row["count"] for row in cursor.fetchall()}

            # モデル別
            cursor.execute("""
                SELECT model_type, COUNT(*) as count
                FROM experiments
                GROUP BY model_type
            """)
            stats["by_model"] = {row["model_type"]: row["count"] for row in cursor.fetchall()}

            # データセット別
            cursor.execute("""
                SELECT dataset_type, COUNT(*) as count
                FROM experiments
                GROUP BY dataset_type
            """)
            stats["by_dataset"] = {row["dataset_type"]: row["count"] for row in cursor.fetchall()}

            return stats
