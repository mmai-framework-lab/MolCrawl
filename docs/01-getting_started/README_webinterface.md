# RIKEN Dataset Foundational Models - Web Interface

This web interface visualizes and manages information about Hugging Face pre-training datasets.

## Overview

You can comprehensively view and manage `training_ready_hf_dataset` data under `learning_source` for `genome_sequence`, `molecule_nat_lang`, `protein_sequence`, and `rna`.

### Main Features

- **Dataset overview**: Dataset statistics across all models
- **Model-specific details**: Detailed Hugging Face dataset info for each model
- **File structure view**: Hierarchical directory and file view
- **Size and stats**: Data size, file count, and update timestamps
- **Real-time updates**: Real-time data retrieval through API endpoints

## Quick Start

### Requirements

- Node.js (v14 or later)
- npm
- Appropriate file access permissions

### 1. Start services

```bash
# Grant execute permission
chmod +x start-new.sh

# Start services
./start-new.sh
```

### 2. Access URLs

- **Frontend**: <http://localhost:3000>
- **API**: <http://localhost:3001>

## Directory Structure

```text
molcrawl-web/
├── api/                          # Backend API
│   ├── server.js                 # Main server
│   ├── routes.js                 # API route definitions
│   ├── dataset-info.js           # Dataset information API
│   ├── directory.js              # Directory structure API
│   └── package.json
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── DatasetInfo.js    # HF dataset display
│   │   │   ├── DatasetInfo.css
│   │   │   ├── DirectoryViewer.js
│   │   │   └── DirectoryViewer.css
│   │   ├── App.js                # Main application
│   │   ├── App.css
│   │   └── index.js
│   ├── public/
│   └── package.json
└── start-new.sh                  # Startup script
```

## API Endpoints

### Hugging Face dataset information

| Endpoint                          | Description                           |
| --------------------------------- | ------------------------------------- |
| `GET /api/datasets/all`           | Dataset information for all models    |
| `GET /api/datasets/:modelName`    | Detailed information for one model    |
| `GET /api/datasets/summary/stats` | Statistical summary                   |

### Directory structure

| Endpoint                    | Description                    |
| --------------------------- | ------------------------------ |
| `GET /api/directory`        | Get directory structure        |
| `GET /api/directory/expand` | Expand a directory             |

## UI Layout

### 1. HF Dataset Tab

#### Overview screen

- **Stat cards**: Total model count, models with data, total file count, total size
- **Model grid**: Per-model detail cards
  - compounds: Chemical compound structures
  - genome_sequence: Genome DNA sequences
  - molecule_nat_lang: Molecule-related natural language descriptions
  - protein_sequence: Protein amino acid sequences
  - rna: RNA nucleotide sequences

#### Model detail screen

- **Basic info**: Dataset count, file count, size, update time
- **Dataset list**: Detailed information for each dataset
- **Sample files**: Example files within each dataset

### 2. Directory Browser Tab

- **Hierarchy view**: Folder/file tree display
- **Expand feature**: Click to expand folder contents
- **Size view**: File and folder size display

## Customization

### Change data source

Change `LEARNING_SOURCE_BASE` in `api/dataset-info.js`:

```javascript
const LEARNING_SOURCE_BASE = "/path/to/your/learning_source";
```

### Add model configuration

Add a new model entry to the `MODELS` object:

```javascript
const MODELS = {
  // Existing models...
  new_model: {
    name: "New Model",
    description: "New model description",
    icon: "",
  },
};
```

## Troubleshooting

### Common issues

1. **Port already in use**

   ```bash
   # Check ports
   lsof -i :3000
   lsof -i :3001

   # Kill process
   kill -9 <PID>
   ```

2. **Permission error**

   ```bash
   # Grant execute permission
   chmod +x start-new.sh

   # Check directory access permission
   ls -la "$LEARNING_SOURCE_DIR"
   ```

3. **Dependency error**

   ```bash
   # Clear cache and reinstall
   cd api && rm -rf node_modules package-lock.json && npm install
   cd ../frontend && rm -rf node_modules package-lock.json && npm install
   ```

### Check logs

```bash
# API log
tail -f logs/api.log

# Frontend log
tail -f logs/frontend.log
```

## Development and Debugging

### Start in development mode

```bash
# API development mode
cd api && npm run dev

# Frontend development mode (separate terminal)
cd frontend && npm start
```

### Debug information

Use the browser developer tools network tab to monitor API call status.

## Performance Optimization

- **Large datasets**: Pagination is recommended
- **Real-time updates**: WebSocket support can enable real-time monitoring
- **Caching**: API response caching can be added with Redis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## License

MIT License

---

**RIKEN Dataset Foundational Models Management Interface**
_Developed for efficient dataset management and visualization_
