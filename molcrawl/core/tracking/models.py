"""
Experiment management data model definition
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ExperimentStatus(str, Enum):
    """Experiment Status"""

    PENDING = "pending"  # Not executed
    RUNNING = "running"  # Running
    COMPLETED = "completed"  # Completed
    FAILED = "failed"  # failed
    CANCELLED = "cancelled"  # cancel
    SKIPPED = "skipped"  # skipped


class ExperimentType(str, Enum):
    """Experiment type"""

    DATA_PREPARATION = "data_preparation"  # Data preparation
    TRAINING = "training"  # model training
    EVALUATION = "evaluation"  # Evaluation
    VISUALIZATION = "visualization"  # visualization
    INFERENCE = "inference"  # inference


class ModelType(str, Enum):
    """Model type"""

    GPT2 = "gpt2"
    BERT = "bert"
    GPN = "gpn"
    OTHER = "other"


class DatasetType(str, Enum):
    """Dataset type"""

    COMPOUNDS = "compounds"
    GENOME_SEQUENCE = "genome_sequence"
    MOLECULE_NAT_LANG = "molecule_related_natural_language"
    PROTEIN_SEQUENCE = "protein_sequence"
    RNA = "rna"
    PROTEINGYM = "proteingym"
    CLINVAR = "clinvar"
    OMIM = "omim"
    COSMIC = "cosmic"
    OTHER = "other"


@dataclass
class ExperimentStep:
    """Each step of the experiment"""

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
    """Experiment Log Entry"""

    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    source: Optional[str] = None  # Log source (file name, function name, etc.)

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
    """Overall information about the experiment"""

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

    # Setting information
    config_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # Results information
    results_dir: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    # steps and logs
    steps: List[ExperimentStep] = field(default_factory=list)
    logs: List[ExperimentLog] = field(default_factory=list)

    # metadata
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
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
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
        data["steps"] = [ExperimentStep.from_dict(step) for step in data.get("steps", [])]
        data["logs"] = [ExperimentLog.from_dict(log) for log in data.get("logs", [])]
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Experiment":
        """Restore from JSON string"""
        return cls.from_dict(json.loads(json_str))
