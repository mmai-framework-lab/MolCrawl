# MolCrawl Dataset Browser

**Fundamental Models Dataset Explorer** for visualizing the MolCrawl dataset directory structure in a browser.

## Overview

This web application provides an interactive view of dataset folders under the project data root, with a focus on directory exploration for day-to-day operations.

## Key Features

- Directory tree view with hierarchical structure
- Expand/collapse folder navigation
- File metadata display (size and item counts)
- Refresh to fetch current directory state
- Responsive layout for desktop and mobile

## Tech Stack

### Frontend

- React.js 19.1.1
- CSS3
- Fetch API

### Backend

- Node.js + Express
- CORS
- Node File System API

## Installation

```bash
cd <PROJECT_ROOT>/molcrawl-web
npm install
```

## Usage

### 1. Development mode (recommended)

```bash
npm run dev
```

Or run the startup script:

```bash
./start.sh
```

### 2. Start services separately

```bash
# backend only (port 3001)
npm run server

# frontend only (port 3000)
npm start
```

### 3. Production mode

```bash
npm run prod
```

## Access URLs

- Web UI: <http://localhost:3000>
- API server: <http://localhost:3001>
- Health check: <http://localhost:3001/api/health>

## API Endpoints

### Directory tree

```text
GET /api/directory?path={directory_path}
```

### Expand child directory

```text
GET /api/directory/expand?path={directory_path}
```

### Health check

```text
GET /api/health
```

## Target Dataset Areas

Commonly explored subdirectories include:

- `cellxgene/`: single-cell RNA sequencing datasets
- `refseq/`: genome sequence datasets (RefSeq)
- `uniprot/`: protein sequence datasets (UniProt)

## Security

- Path validation limits access to allowed base directories
- CORS policy is restricted for local development usage
- Read-only file operations (no write endpoints)

## Quick Start

```bash
cd molcrawl-web
npm install
./start.sh
# open http://localhost:3000
```
