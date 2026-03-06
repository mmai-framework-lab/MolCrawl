# Experiment Tracking - Quick Start

## 1. Initial Setup

```bash
chmod +x workflows/setup_experiment_system.sh
./workflows/setup_experiment_system.sh
```

This setup installs required dependencies and prepares the local environment.

## 2. Start Services

```bash
./workflows/start_experiment_system.sh
```

Or run separately:

```bash
python workflows/start_api_server.py
cd molcrawl-web
npm run dev
```

## 3. Open the UI

- Dashboard: <http://localhost:3000>
- API Docs: <http://localhost:8000/docs>

## 4. Run a Sample

```bash
python misc/experiment_tracker_sample.py
```

## 5. Verify Installation

```bash
python molcrawl/debug/test_experiment_system.py
```

## Troubleshooting

### Port already in use

```bash
lsof -i :8000
lsof -i :3000
kill -9 <PID>
```

### Reset local database

```bash
rm -f experiment_data/experiments.db
```

## Next Docs

For full details:
- [EXPERIMENT_TRACKING_README.md](EXPERIMENT_TRACKING_README.md)
- [EXPERIMENT_TRACKING_ARCHITECTURE.md](EXPERIMENT_TRACKING_ARCHITECTURE.md)
