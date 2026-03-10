# MolCrawl Web

MolCrawl Dataset Browser - A web interface for exploring datasets

## Requirements

- Node.js 18 or higher
- npm
- Python 3.x
- **`LEARNING_SOURCE_DIR` environment variable must be set (required)**

## Quick Start

### 1. Install dependencies

```bash
cd molcrawl-web
npm install
```

### 2. Check available directories

```bash
npm run check-env
```

If an error is shown, the available `learning_source` directories will be listed.

### 3. Start the server

**Recommended method (easy start script)**:

```bash
# Start on default ports (3000, 3001)
LEARNING_SOURCE_DIR="learning_source_20251210" ./start-dev.sh

# Start on custom ports (frontend=8090, API=8091)
LEARNING_SOURCE_DIR="learning_source_20251210" ./start-dev.sh 8090 8091
```

**Or specify environment variables directly**:

```bash
# Set environment variables and start
LEARNING_SOURCE_DIR="learning_source_20251210" PORT=8090 API_PORT=8091 npm run dev
```

**Manual start (in separate terminals)**:

```bash
# Terminal 1: Backend API
LEARNING_SOURCE_DIR="learning_source_20251210" API_PORT=8091 npm run server

# Terminal 2: Frontend (set REACT_APP_API_PORT to specify proxy target)
PORT=8090 REACT_APP_API_PORT=8091 npm start
```

**Important**:

- Use `API_PORT` for the backend API server
- Use `PORT` and `REACT_APP_API_PORT` for the frontend
- Make sure both servers are running on different ports

Default ports:

- Backend API: port 3001
- Frontend: port 3000

Health check:

```bash
# Check backend
curl http://localhost:3001/api/health

# Check ports
lsof -i :3001  # Backend
lsof -i :3000  # Frontend
```

#### Changing port numbers

By default, backend uses port 3001 and frontend uses port 3000.
If a port is already in use, you can change it with the following methods.

**Method 1: Specify via environment variables (recommended)**

```bash
# Dev mode: start frontend on 8090, backend on 8091
LEARNING_SOURCE_DIR="learning_source_202508" PORT=8090 API_PORT=8091 npm run dev

# Or run in separate terminals
# Terminal 1: Start backend on 8091
LEARNING_SOURCE_DIR="learning_source_202508" API_PORT=8091 npm run server

# Terminal 2: Start frontend on 8090 (automatically proxies to API on 8091)
PORT=8090 API_PORT=8091 npm start
```

**Method 2: Specify via command-line argument (backend only)**

```bash
# Specify backend port
LEARNING_SOURCE_DIR="learning_source_202508" node server.js --port 8091
# or
LEARNING_SOURCE_DIR="learning_source_202508" node server.js -p 8091
```

**Important**:

- `PORT`: Port number for the React development server (frontend)
- `API_PORT`: Port number for the Express API server (backend)
- Do not set both to the same port (they will conflict)

**Show help**

```bash
node server.js --help
```

### 4. Access in browser

- **Frontend**: <http://localhost:3000>
- **Backend API**: <http://localhost:3001/api/health>

## NPM Scripts

### Development

- `npm run dev` - Start frontend and backend simultaneously (recommended)
- `npm run dev:nfs` - For NFS mount environments (polling enabled)
- `npm start` - Start frontend only
- `npm run server` - Start backend only
- `npm run check-env` - Check environment variables and configuration

### Build & Test

- `npm run build` - Production build
- `npm test` - Run tests
- `npm run prod` - Start server after build
- `npm run prod:serve` - Serve built files (stable on NFS)
- `npm run serve:build` - Serve build directory as static files

### Code Quality

- `npm run lint` - Check code with ESLint
- `npm run lint:fix` - Auto-fix with ESLint
- `npm run format` - Format with Prettier
- `npm run format:check` - Check formatting only

## Starting in NFS Mount Environments

Special configuration is required when running the project on an NFS-mounted directory (e.g., `/wren`).

### Problem and Cause

`inotify` (file change monitoring) does not work correctly on NFS file systems, which can prevent webpack-dev-server from working normally.

### Solutions

#### Method 1: Use start-dev.sh (recommended — auto-detects NFS)

```bash
# Auto-detects NFS mount and enables polling mode
LEARNING_SOURCE_DIR="learning_source_20251210" ./start-dev.sh 9090 9091
```

#### Method 2: Use npm run dev:nfs

```bash
LEARNING_SOURCE_DIR="learning_source_20251210" PORT=9090 API_PORT=9091 npm run dev:nfs
```

#### Method 3: Explicitly set environment variables

```bash
CHOKIDAR_USEPOLLING=true WATCHPACK_POLLING=true \
LEARNING_SOURCE_DIR="learning_source_20251210" PORT=9090 API_PORT=9091 npm run dev
```

#### Method 4: Use a Production Build (most stable)

```bash
# Build
npm run build

# Start API server + static file serving
LEARNING_SOURCE_DIR="learning_source_20251210" API_PORT=9091 npm run prod:serve -- -l 9090
```

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for more details.

## Environment Variables

### LEARNING_SOURCE_DIR (required)

Specifies the root directory for datasets.

```bash
export LEARNING_SOURCE_DIR="learning_source_202508"
```

Available directories:

- `learning_source_202508`
- `learning_source_20251006_genome_all`
- `learning_source_20251020-molecule-nl`

### API_PORT (optional)

**Specifies the backend API server port number (default: 3001).**

```bash
export API_PORT=8091
```

### PORT (optional)

**Specifies the React development server (frontend) port number (default: 3000).**

```bash
export PORT=8090
```

Notes:

- The backend server uses `API_PORT` preferentially
- A value specified via `--port` command-line argument takes highest priority
- Set different port numbers for the frontend and backend

### To set persistently

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export LEARNING_SOURCE_DIR="learning_source_202508"
export PORT=8090       # For frontend
export API_PORT=8091   # For backend API
```

Apply the settings:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for details.

### Common Issues

#### Server won't start

```bash
❌ ERROR: LEARNING_SOURCE_DIR environment variable is required!
```

**Solution**: Set the environment variable

```bash
LEARNING_SOURCE_DIR="learning_source_202508" npm run dev
```

#### Proxy error: ECONNREFUSED

```
Proxy error: Could not proxy request /api/... from localhost:XXXXX to http://localhost:3001.
(ECONNREFUSED)
```

**Cause**: The backend server (port 3001) is not running

**Solution**:

1. Check if the backend is running

   ```bash
   lsof -i :3001
   ```

2. If not running, start it in a separate terminal

   ```bash
   cd molcrawl-web
   LEARNING_SOURCE_DIR="learning_source_202508" npm run server
   ```

3. Or use `./start-both.sh`

   ```bash
   export LEARNING_SOURCE_DIR="learning_source_202508"
   ./start-both.sh
   ```

4. When using `npm run dev`, `concurrently` should start both, but if an error occurs, start them manually

#### Port already in use

```bash
Error: listen EADDRINUSE: address already in use :::3001
```

**Solution**: Specify a different port number

```bash
# Start frontend and backend on different ports
LEARNING_SOURCE_DIR="learning_source_202508" PORT=8090 API_PORT=8091 npm run dev

# Or start individually
# Change backend only
API_PORT=8091 LEARNING_SOURCE_DIR="learning_source_202508" npm run server

# Change frontend only
PORT=8090 npm start
```

**Important**: Do not set `PORT` and `API_PORT` to the same value. The React development server and API server will conflict.

#### 500 errors occur

The backend server may not be running.

**Solution**: Use `npm run dev` or `./start-both.sh` instead of `npm start`

## Features

### 📊 Dataset Preparation Progress Monitoring

Monitor the progress of five dataset preparation scripts in real time:

- **Protein Sequence (Uniprot)** - 3 steps
  - Uniprot Download → FASTA to Raw → Tokenization
- **Genome Sequence (RefSeq)** - 4 steps
  - RefSeq Download → FASTA to Raw → Tokenizer Training → Raw to Parquet
- **RNA (CellxGene)** - 5 steps
  - Build List → Download → H5AD to Loom → Tokenization → Vocabulary
- **Molecule NL (SMolInstruct)** - 2 steps
  - Dataset Download/Copy → Tokenization & Processing
- **Compounds (OrganiX13)** - 3 steps
  - OrganiX13 Download → SMILES & Scaffolds Tokenization → Statistics

The progress status of each dataset is automatically determined by the existence of marker files and output files.

#### 🚀 Preparation Script Execution (New Feature)

You can directly run preparation scripts from the "Preparation Progress" card of each dataset:

- **Phase 01 button**: Runs data download and basic preprocessing scripts
  - e.g. `01-protein_sequence-prepare.sh`
- **Phase 02 button**: Runs GPT-2 dataset preparation scripts
  - e.g. `02-protein_sequence-prepare-gpt2.sh`

**Feature details**:

- ✅ Start script execution with one click
- 📋 Display logs in real-time in a modal (auto-refresh every 2 seconds)
- ⏹️ Stop a running script
- 📊 Show execution status (PID, elapsed time, status)
- 🔄 Automatically refresh progress after script completion

**How to use**:

1. Check the "Preparation Progress" card in each dataset tab
2. Click the "▶ Phase 01" or "▶ Phase 02" button
3. A log modal opens and displays logs in real time
4. Wait for the script to complete, or click "⏹ Stop" to abort
5. Closing the modal does not stop the script — it continues running in the background

#### How to Use (Progress Check)

1. Access <http://localhost:3000> in a web browser
2. Click the "Preparation" tab
3. Check the progress of each dataset
4. Use the auto-refresh option to refresh every 5 seconds

## Project Structure

```
molcrawl-web/
├── api/                    # Backend API
│   ├── directory.js       # Directory API
│   ├── dataset-progress.js # Dataset preparation progress API
│   ├── genome-species.js  # Genome species API
│   └── zinc-checker.js    # ZINC20 data checker
├── src/                    # React frontend
│   ├── App.js             # Main application
│   ├── DatasetProgress.js # Dataset preparation progress component
│   ├── ExperimentDashboard.js
│   ├── GenomeSpeciesList.js
│   └── ZincChecker.js
├── public/                 # Static files
├── server.js              # Express server
├── package.json           # Dependencies
└── check-config.js        # Configuration check script
```

## API Endpoints

### Health Check

- `GET /api/health` - Server status

### Directory Operations

- `GET /api/directory` - Get root directory structure
- `GET /api/directory/expand?path=<path>` - Expand a directory
- `GET /api/directory/tree?maxDepth=5` - Get full tree

### Genome Data

- `GET /api/genome/species` - Get genome species list
- `GET /api/genome/species/category?category=<category>` - Species list by category

### ZINC20 Data

- `GET /api/zinc/check` - Check ZINC20 data
- `GET /api/zinc/count` - Get ZINC20 data count

### Dataset Preparation Progress

- `GET /api/dataset-progress` - Get preparation progress for all datasets
- `GET /api/dataset-progress/:datasetKey` - Get detailed progress for a specific dataset
  - `datasetKey`: `protein_sequence`, `genome_sequence`, `rna`, `molecule_nat_lang`, `compounds`

### Preparation Script Execution (New Feature)

- `GET /api/preparation-runner/scripts` - List available scripts
- `POST /api/preparation-runner/start` - Run a preparation script
  - Body: `{ dataset: 'protein_sequence', phase: 'phase01' }`
- `GET /api/preparation-runner/status/:dataset/:phase` - Get execution status
- `GET /api/preparation-runner/log/:dataset/:phase?lines=200` - Get execution log
- `POST /api/preparation-runner/stop` - Stop a script
  - Body: `{ dataset: 'protein_sequence', phase: 'phase01' }`
- `GET /api/preparation-runner/all-status` - Get all execution statuses

## Development

### ESLint Configuration

Configured in `.eslintrc.json`. See [ESLINT_SETUP.md](./ESLINT_SETUP.md) for details.

### Prettier Configuration

Configured in `.prettierrc.json`.
