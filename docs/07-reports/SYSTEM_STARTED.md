# Experiment Tracking System - Startup Completion Report

## System Startup Status

The experiment tracking system started successfully.

### Running Services

1. **API Server** (FastAPI + Uvicorn)
   - URL: <http://localhost:8000>
   - API Docs: <http://localhost:8000/docs>
   - Status: Running
   - Database: `experiment_data/experiments.db`
   - Recorded experiments: 8

2. **Web Frontend** (React + Express)
   - URL: <http://localhost:3000>
   - Backend API: <http://localhost:3001>
   - Status: Running

### Current Experiment Summary

- Total experiments: 8
- Completed: 7
- Failed: 1

By type:

- Data preparation: 2
- Evaluation: 5
- Training: 1

## Access

Open:

```text
http://localhost:3000
```

Then select the **Experiments** tab.

## Available Operations

1. View experiment list with timeline and filters
2. Open experiment details (steps, metrics, logs)
3. Monitor real-time statistics (10-second updates)

## Integration Example

```python
from molcrawl.core.tracking.helpers import experiment_context
from molcrawl.core.tracking import ExperimentType, ModelType, DatasetType

with experiment_context(
    name="GPT2 ProteinGym Training",
    experiment_type=ExperimentType.TRAINING,
    model_type=ModelType.GPT2,
    dataset_type=DatasetType.PROTEINGYM
) as exp:
    exp.start_step("data", "Load data")
    # your code
    exp.complete_step("data")
    exp.add_metric("accuracy", 0.95)
```

## Service Management

### Health checks

```bash
curl http://localhost:8000/api/statistics
curl http://localhost:3000
```

### Stop services

```bash
pkill -f "start_api_server.py"
pkill -f "npm run dev"
```

### Restart services

```bash
cd <PROJECT_ROOT>

# API server
source miniconda/bin/activate conda
PYTHONPATH=$PWD:$PYTHONPATH nohup python start_api_server.py > logs/api_server.log 2>&1 &

# Web frontend
cd molcrawl-web
nohup npm run dev > ../logs/web_frontend.log 2>&1 &
```

## Related Docs

- `05-experiment_tracking/EXPERIMENT_TRACKING_QUICKSTART.md`
- `05-experiment_tracking/EXPERIMENT_TRACKING_README.md`
- `05-experiment_tracking/EXPERIMENT_TRACKING_ARCHITECTURE.md`
- `05-experiment_tracking/EXPERIMENT_TRACKING_SUMMARY.md`

## Notes

- Logs are stored in `logs/`
- Database file: `experiment_data/experiments.db`
