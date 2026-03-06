# Experiment Tracking System

## Overview

This system centralizes experiment progress, logs, and results for training/evaluation workflows in this repository.
It is built for local or self-managed usage with lightweight components.

## Main Features

- Automatic experiment lifecycle tracking (start/end/status/duration)
- Step-level tracking inside each experiment
- Metric and result recording
- Centralized logging
- Web dashboard + API access
- SQLite-based storage (no external DB required)

## Project Structure

```text
riken-dataset-fundational-model/
├── molcrawl/experiment_tracker/
│   ├── __init__.py
│   ├── models.py
│   ├── database.py
│   ├── tracker.py
│   ├── helpers.py
│   └── api.py
├── molcrawl-web/
├── workflows/
│   ├── start_api_server.py
│   ├── start_experiment_system.sh
│   └── setup_experiment_system.sh
├── molcrawl/debug/test_experiment_system.py
├── misc/experiment_tracker_sample.py
└── experiment_data/
    └── experiments.db
```

## Quick Start

### 1. Install dependencies

```bash
pip install fastapi uvicorn sqlalchemy
cd molcrawl-web
npm install
cd ..
```

### 2. Start the system

Recommended:

```bash
chmod +x workflows/start_experiment_system.sh
./workflows/start_experiment_system.sh
```

Or run separately:

```bash
python workflows/start_api_server.py
cd molcrawl-web
npm run dev
```

### 3. Access

- Web UI: <http://localhost:3000>
- API docs: <http://localhost:8000/docs>

## Usage Patterns

### Pattern 1: Context manager (recommended)

```python
from molcrawl.experiment_tracker.helpers import experiment_context
from molcrawl.experiment_tracker import ExperimentType, ModelType, DatasetType

with experiment_context(
    name="GPT2 ProteinGym Training",
    experiment_type=ExperimentType.TRAINING,
    model_type=ModelType.GPT2,
    dataset_type=DatasetType.PROTEINGYM,
    config={"epochs": 10, "batch_size": 32},
) as exp:
    exp.start_step("data_loading", "Load dataset")
    # load_data()
    exp.complete_step("data_loading", output_path="data/processed/train.pt")

    exp.start_step("training", "Train model")
    # train_model()
    exp.complete_step("training", output_path="models/gpt2_proteingym.pt")

    exp.add_metric("accuracy", 0.95)
    exp.add_result("model_path", "models/gpt2_proteingym.pt")
```

### Pattern 2: Decorator

```python
from molcrawl.experiment_tracker.helpers import track_experiment
from molcrawl.experiment_tracker import ExperimentType, ModelType, DatasetType

@track_experiment(
    name="BERT ClinVar Evaluation",
    experiment_type=ExperimentType.EVALUATION,
    model_type=ModelType.BERT,
    dataset_type=DatasetType.CLINVAR,
)
def run_evaluation(config):
    return {"accuracy": 0.92, "f1_score": 0.89}
```

### Pattern 3: Manual tracking API

```python
from molcrawl.experiment_tracker import ExperimentTracker, ExperimentType, ModelType, DatasetType

tracker = ExperimentTracker()
exp_id = tracker.start_experiment(
    name="RNA Data Preparation",
    experiment_type=ExperimentType.DATA_PREPARATION,
    model_type=ModelType.GPT2,
    dataset_type=DatasetType.RNA,
)

tracker.start_step(exp_id, "preprocess", "Tokenize data")
# preprocess_data()
tracker.complete_step(exp_id, "preprocess")
tracker.complete_experiment(exp_id, metrics={"num_samples": 100000})
```

## Basic Validation

```bash
python molcrawl/debug/test_experiment_system.py
python misc/experiment_tracker_sample.py
```

## Related Docs

- [EXPERIMENT_TRACKING_QUICKSTART.md](EXPERIMENT_TRACKING_QUICKSTART.md)
- [EXPERIMENT_TRACKING_ARCHITECTURE.md](EXPERIMENT_TRACKING_ARCHITECTURE.md)
- [EXPERIMENT_TRACKING_SUMMARY.md](EXPERIMENT_TRACKING_SUMMARY.md)
