"""
実験管理のデータモデル定義
"""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
import json


class ExperimentStatus(str, Enum):
    """実験ステータス"""

    PENDING = "pending"  # 未実行
    RUNNING = "running"  # 実行中
    COMPLETED = "completed"  # 完了
    FAILED = "failed"  # 失敗
    CANCELLED = "cancelled"  # キャンセル
    SKIPPED = "skipped"  # スキップ


class ExperimentType(str, Enum):
    """実験タイプ"""

    DATA_PREPARATION = "data_preparation"  # データ準備
    TRAINING = "training"  # モデル訓練
    EVALUATION = "evaluation"  # 評価
    VISUALIZATION = "visualization"  # 可視化
    INFERENCE = "inference"  # 推論


class ModelType(str, Enum):
    """モデルタイプ"""

    GPT2 = "gpt2"
    BERT = "bert"
    GPN = "gpn"
    OTHER = "other"


class DatasetType(str, Enum):
    """データセットタイプ"""

    COMPOUNDS = "compounds"
    GENOME_SEQUENCE = "genome_sequence"
    MOLECULE_NL = "molecule_related_natural_language"
    PROTEIN_SEQUENCE = "protein_sequence"
    RNA = "rna"
    PROTEINGYM = "proteingym"
    CLINVAR = "clinvar"
    OMIM = "omim"
    COSMIC = "cosmic"
    OTHER = "other"


@dataclass
class ExperimentStep:
    """実験の各ステップ"""

    step_id: str
    step_name: str
    status: ExperimentStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    command: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["start_time"] = self.start_time.isoformat() if self.start_time else None
        data["end_time"] = self.end_time.isoformat() if self.end_time else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentStep":
        data["status"] = ExperimentStatus(data["status"])
        if data.get("start_time"):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        return cls(**data)


@dataclass
class ExperimentLog:
    """実験ログエントリ"""

    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    source: Optional[str] = None  # ログのソース（ファイル名、関数名など）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentLog":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Experiment:
    """実験の全体情報"""

    experiment_id: str
    experiment_name: str
    experiment_type: ExperimentType
    model_type: ModelType
    dataset_type: DatasetType
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None

    # 設定情報
    config_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # 結果情報
    results_dir: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    # ステップとログ
    steps: List[ExperimentStep] = field(default_factory=list)
    logs: List[ExperimentLog] = field(default_factory=list)

    # メタデータ
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type.value,
            "model_type": self.model_type.value,
            "dataset_type": self.dataset_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "total_duration_seconds": self.total_duration_seconds,
            "config_path": self.config_path,
            "config": self.config,
            "results_dir": self.results_dir,
            "results": self.results,
            "metrics": self.metrics,
            "steps": [step.to_dict() for step in self.steps],
            "logs": [log.to_dict() for log in self.logs],
            "tags": self.tags,
            "notes": self.notes,
            "environment": self.environment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        data["experiment_type"] = ExperimentType(data["experiment_type"])
        data["model_type"] = ModelType(data["model_type"])
        data["dataset_type"] = DatasetType(data["dataset_type"])
        data["status"] = ExperimentStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        data["steps"] = [
            ExperimentStep.from_dict(step) for step in data.get("steps", [])
        ]
        data["logs"] = [ExperimentLog.from_dict(log) for log in data.get("logs", [])]
        return cls(**data)

    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Experiment":
        """JSON文字列から復元"""
        return cls.from_dict(json.loads(json_str))
