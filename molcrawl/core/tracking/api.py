"""
FastAPI-based experiment management API
Works as a backend for molcrawl-web
"""

from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
except ModuleNotFoundError:
    FastAPI = None
    HTTPException = Exception
    Query = None
    CORSMiddleware = None

from molcrawl.core.tracking import (
    DatasetType,
    ExperimentStatus,
    ExperimentTracker,
    ExperimentType,
    ModelType,
)

# add project root to path
project_root = Path(__file__).parent.parent.parent.parent

# global tracker instance
tracker = ExperimentTracker()

if FastAPI is not None:
    app = FastAPI(
        title="MolCrawl Experiment Management API",
        description="Experiment management system API",
        version="1.0.0",
    )

    # CORS settings (FastAPI standard CORS Middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Specific origins should be specified in production environments
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """APIinformation"""
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
        status: Optional[str] = Query(None, description="Filter by status"),
        experiment_type: Optional[str] = Query(None, description="Filter by experiment type"),
        model_type: Optional[str] = Query(None, description="Filter by model type"),
        dataset_type: Optional[str] = Query(None, description="Filter by dataset type"),
        limit: int = Query(100, description="Number of retrieved results"),
        offset: int = Query(0, description="offset"),
    ):
        """Get experiment list"""
        try:
            # Convert parameter to Enum
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
        """Get experiment details"""
        experiment = tracker.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        return experiment.to_dict()

    @app.get("/api/statistics")
    async def get_statistics():
        """Get statistics"""
        return tracker.get_statistics()

    @app.get("/api/experiments/{experiment_id}/logs")
    async def get_experiment_logs(
        experiment_id: str,
        level: Optional[str] = Query(None, description="Filter by log level"),
        limit: int = Query(1000, description="Number of retrieved results"),
    ):
        """Get experiment log"""
        experiment = tracker.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        logs = experiment.logs

        # filter by level
        if level:
            logs = [log for log in logs if log.level == level.upper()]

        # limit
        logs = logs[-limit:]

        return {
            "experiment_id": experiment_id,
            "total": len(logs),
            "logs": [log.to_dict() for log in logs],
        }

    @app.get("/api/experiments/{experiment_id}/steps")
    async def get_experiment_steps(experiment_id: str):
        """Get a list of experiment steps"""
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
        """Health Check"""
        return {"status": "healthy", "database": "connected"}
else:
    app = None


if __name__ == "__main__":
    import uvicorn

    if app is None:
        raise RuntimeError("FastAPI is not available. Install fastapi to run the API server.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
