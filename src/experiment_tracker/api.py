"""
FastAPI ベースの実験管理API
molcrawl-webのバックエンドとして動作
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pathlib import Path
import sys
from src.experiment_tracker import (
    ExperimentTracker,
    ExperimentStatus,
    ExperimentType,
    ModelType,
    DatasetType,
)

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

app = FastAPI(
    title="MolCrawl Experiment Management API",
    description="実験管理システムのAPI",
    version="1.0.0",
)

# CORS設定（FastAPI標準のCORSMiddleware）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では具体的なオリジンを指定すべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバルトラッカーインスタンス
tracker = ExperimentTracker()


@app.get("/")
async def root():
    """API情報"""
    return {
        "name": "MolCrawl Experiment Management API",
        "version": "1.0.0",
        "endpoints": {
            "experiments": "/api/experiments",
            "statistics": "/api/statistics",
            "experiment_detail": "/api/experiments/{experiment_id}",
        },
    }


@app.get("/api/experiments")
async def list_experiments(
    status: Optional[str] = Query(None, description="ステータスでフィルタ"),
    experiment_type: Optional[str] = Query(None, description="実験タイプでフィルタ"),
    model_type: Optional[str] = Query(None, description="モデルタイプでフィルタ"),
    dataset_type: Optional[str] = Query(None, description="データセットタイプでフィルタ"),
    limit: int = Query(100, description="取得件数"),
    offset: int = Query(0, description="オフセット"),
):
    """実験一覧を取得"""
    try:
        # パラメータをEnumに変換
        status_enum = ExperimentStatus(status) if status else None
        exp_type_enum = ExperimentType(experiment_type) if experiment_type else None
        model_type_enum = ModelType(model_type) if model_type else None
        dataset_type_enum = DatasetType(dataset_type) if dataset_type else None

        experiments = tracker.list_experiments(
            status=status_enum,
            experiment_type=exp_type_enum,
            model_type=model_type_enum,
            dataset_type=dataset_type_enum,
            limit=limit,
            offset=offset,
        )

        return {
            "total": len(experiments),
            "experiments": [exp.to_dict() for exp in experiments],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """実験の詳細を取得"""
    experiment = tracker.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return experiment.to_dict()


@app.get("/api/statistics")
async def get_statistics():
    """統計情報を取得"""
    return tracker.get_statistics()


@app.get("/api/experiments/{experiment_id}/logs")
async def get_experiment_logs(
    experiment_id: str,
    level: Optional[str] = Query(None, description="ログレベルでフィルタ"),
    limit: int = Query(1000, description="取得件数"),
):
    """実験のログを取得"""
    experiment = tracker.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    logs = experiment.logs

    # レベルでフィルタ
    if level:
        logs = [log for log in logs if log.level == level.upper()]

    # 制限
    logs = logs[-limit:]

    return {
        "experiment_id": experiment_id,
        "total": len(logs),
        "logs": [log.to_dict() for log in logs],
    }


@app.get("/api/experiments/{experiment_id}/steps")
async def get_experiment_steps(experiment_id: str):
    """実験のステップ一覧を取得"""
    experiment = tracker.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_id": experiment_id,
        "total": len(experiment.steps),
        "steps": [step.to_dict() for step in experiment.steps],
    }


@app.get("/api/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "database": "connected"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
