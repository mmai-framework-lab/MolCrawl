# Experiment Tracking - Architecture

## Goal

Provide a lightweight and extensible system to track ML experiment execution, steps, logs, and metrics in one place.

## Components

### Core Python Module

Location: `molcrawl/experiment_tracker/`

- `models.py`: Experiment/Step/Log models and enums
- `database.py`: SQLite access layer
- `tracker.py`: tracking API implementation
- `helpers.py`: context manager / decorator helpers
- `api.py`: FastAPI endpoints

### Web UI

Location: `molcrawl-web/`

- React-based dashboard for listing and inspecting experiments

### Runtime Scripts

Location: `workflows/`

- `setup_experiment_system.sh`
- `start_experiment_system.sh`
- `start_api_server.py`

## Data Model

### Experiment

Stores metadata and lifecycle fields:
- id, name, type, model_type, dataset_type
- status (`pending`, `running`, `completed`, `failed`)
- timestamps (created/started/completed)
- config, results, metrics, tags

### ExperimentStep

Stores per-step execution data:
- step_name, status
- start/end timestamps
- command, output_path, error_message

### ExperimentLog

Stores logs with:
- timestamp
- level
- message
- source

## Storage

- Local SQLite DB: `experiment_data/experiments.db`
- Suitable for local and small team workflows
- Easy backup/restore by copying DB file

## API Layer

FastAPI (`molcrawl/experiment_tracker/api.py`) provides endpoints for:
- listing experiments
- reading details
- filtering by status/type/model/dataset
- reading metrics and step history

## Evaluation Output Integration

Evaluation scripts use a structured output policy under `LEARNING_SOURCE_DIR`, and utility code is centralized at:

- `molcrawl/utils/evaluation_output.py`

Related scripts include:
- `molcrawl/tasks/evaluation/proteingym/gpt2_evaluation.py`
- `molcrawl/tasks/evaluation/clinvar/gpt2_evaluation.py`
- `molcrawl/tasks/evaluation/cosmic/gpt2_evaluation.py`
- `molcrawl/tasks/evaluation/omim/gpt2_evaluation.py`
- `molcrawl/tasks/evaluation/protein_classification/gpt2_evaluation.py`
- `molcrawl/tasks/evaluation/proteingym/bert_evaluation.py`
- `molcrawl/tasks/evaluation/clinvar/bert_evaluation.py`

## Extension Points

- Add new experiment/model/dataset enums in `models.py`
- Add custom metrics in tracker calls
- Add external notification hooks (Slack, email) on completion events
- Extend API routes for project-specific dashboard features
