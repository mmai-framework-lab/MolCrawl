# MolCrawl Web - Troubleshooting Guide

## Required: Setting Environment Variables

**Important**: This application requires the `LEARNING_SOURCE_DIR` environment variable. It will not start without it.

## How to Start

### Step 1: Check available directories

```bash
cd molcrawl-web
npm run check-env
```

This will display the available `learning_source` directories.

### Step 2: Set the environment variable and start

#### Method 1: Start in a single line (recommended)

```bash
cd molcrawl-web
LEARNING_SOURCE_DIR="learning_source_202508" npm run dev
```

#### Method 2: Export environment variable and start

```bash
cd molcrawl-web
export LEARNING_SOURCE_DIR="learning_source_202508"
npm run dev
```

#### Method 3: Add to .bashrc or .zshrc (persistent)

```bash
# Add to ~/.bashrc or ~/.zshrc
export LEARNING_SOURCE_DIR="learning_source_202508"

# Apply settings
source ~/.bashrc  # or source ~/.zshrc
```

---

## Starting in NFS Mount Environments

### Problem Overview

When running the project on an NFS-mounted directory (e.g., `/wren`), webpack-dev-server may not work correctly.

**Symptoms:**

- Running `npm run dev` shows "Compiled successfully!" but the port is not listening
- `ss -tlnp | grep 9090` shows no port
- `curl http://localhost:9090` returns a "Connection refused" error

**Cause:**

- `inotify` (file change monitoring) does not work correctly on NFS file systems
- webpack-dev-server uses inotify internally to monitor file changes
- If inotify does not work, the HTTP server may not start correctly

### Solutions

#### Method 1: Use the start-dev.sh script (recommended)

The `start-dev.sh` script automatically detects NFS mounts and enables the required settings.

```bash
cd molcrawl-web
LEARNING_SOURCE_DIR="learning_source_20251210" ./start-dev.sh 9090 9091
```

#### Method 2: Set environment variables manually

Start with polling mode enabled:

```bash
cd molcrawl-web
CHOKIDAR_USEPOLLING=true WATCHPACK_POLLING=true \
LEARNING_SOURCE_DIR="learning_source_20251210" \
PORT=9090 API_PORT=9091 npm run dev
```

#### Method 3: Use npm run dev:nfs

An NFS-compatible script is available in package.json:

```bash
cd molcrawl-web
LEARNING_SOURCE_DIR="learning_source_20251210" \
PORT=9090 API_PORT=9091 npm run dev:nfs
```

#### Method 4: Use a Production Build

Serve pre-built files instead of the development server (no hot reload):

```bash
cd molcrawl-web

# Build (run once initially or after changes)
npm run build

# Start API server and frontend simultaneously
LEARNING_SOURCE_DIR="learning_source_20251210" \
API_PORT=9091 npm run prod:serve -- -l 9090
```

Or start separately:

```bash
# Terminal 1: API server
LEARNING_SOURCE_DIR="learning_source_20251210" API_PORT=9091 npm run server

# Terminal 2: Frontend (static file serving)
npx serve build -l 9090
```

### NFS Environment Config Files

`.env.development` contains the following settings:

```bash
# NFS Mount Environment Settings
CHOKIDAR_USEPOLLING=true
WATCHPACK_POLLING=true
```

These settings are loaded automatically, but in some environments you may need to pass them explicitly as environment variables.

### How to check if you are in an NFS environment

Check whether the current directory is NFS-mounted:

```bash
df -T .
```

If the output contains `nfs` or `nfs4`, it is an NFS mount.

---

## Common Errors and Solutions

### Error: `LEARNING_SOURCE_DIR environment variable is required!`

The environment variable is not set.

**Solution**:

```bash
# Check available directories
npm run check-env

# Set env variable and start
LEARNING_SOURCE_DIR="learning_source_202508" npm run dev
```

### Error: `Specified LEARNING_SOURCE_DIR does not exist!`

The specified directory does not exist.

**Solution**:

```bash
# Check the correct directory name
ls -d ../learning_source*

# Restart with the correct name
LEARNING_SOURCE_DIR="correct_directory_name" npm run dev
```

### Error: `ECONNREFUSED localhost:3001`

The backend server is not running.

**Solution**:

- Use `npm run dev` instead of `npm start`
- Or run `npm run server` in a separate terminal

### Port already in use

```bash
# Check the process using port 3001
lsof -i :3001

# Kill the process
kill -9 <PID>

# Or free ports all at once
fuser -k 3000/tcp 3001/tcp
```

### Development server starts but port is not listening

This is an NFS mount environment issue. See the "Starting in NFS Mount Environments" section above.

---

## Access URLs

After starting, the following URLs are available:

- **Frontend**: <http://localhost:3000>
- **Backend API**: <http://localhost:3001/api/health>
- **Directory API**: <http://localhost:3001/api/directory>
