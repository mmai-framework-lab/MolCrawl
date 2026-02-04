#!/bin/bash
# MolCrawl Web Development Server Startup Script
# Usage: ./start-dev.sh [frontend_port] [api_port]
#
# Examples:
#   ./start-dev.sh                    # Use default ports (3000, 3001)
#   ./start-dev.sh 8090 8091         # Use custom ports
#
# NFS Mount Environment:
#   This script automatically detects NFS-mounted filesystems and enables
#   polling mode for file watching, which is required on NFS.

set -e  # Exit on error

# Default ports
FRONTEND_PORT=${1:-3000}
API_PORT=${2:-3001}

# Check if LEARNING_SOURCE_DIR is set
if [ -z "$LEARNING_SOURCE_DIR" ]; then
    echo ""
    echo "❌ ERROR: LEARNING_SOURCE_DIR environment variable is required!"
    echo ""
    echo "Please set it before running this script:"
    echo "  export LEARNING_SOURCE_DIR=\"learning_source_202508\""
    echo ""
    echo "Or run with inline environment variable:"
    echo "  LEARNING_SOURCE_DIR=\"learning_source_202508\" ./start-dev.sh"
    echo ""
    exit 1
fi

# Detect NFS mount
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IS_NFS=false

if df -T "$SCRIPT_DIR" 2>/dev/null | grep -qE "nfs|cifs"; then
    IS_NFS=true
elif mount | grep -q "$(df "$SCRIPT_DIR" 2>/dev/null | tail -1 | awk '{print $1}')" | grep -qE "nfs|cifs"; then
    IS_NFS=true
fi

echo ""
echo "=========================================="
echo "🚀 Starting MolCrawl Web Development Servers"
echo "=========================================="
echo "Learning Source: $LEARNING_SOURCE_DIR"
echo "Frontend Port:   $FRONTEND_PORT"
echo "API Port:        $API_PORT"

if [ "$IS_NFS" = true ]; then
    echo "Filesystem:      NFS (polling mode enabled)"
else
    echo "Filesystem:      Local"
fi

echo "=========================================="
echo ""

# Export environment variables for both servers
export PORT=$FRONTEND_PORT
export API_PORT=$API_PORT
export REACT_APP_API_PORT=$API_PORT
export BROWSER=none

# Enable polling mode for NFS environments
# This is required because inotify does not work properly on NFS
if [ "$IS_NFS" = true ]; then
    echo "📂 NFS mount detected. Enabling polling mode for file watching..."
    echo ""
    export CHOKIDAR_USEPOLLING=true
    export WATCHPACK_POLLING=true
fi

# Start both servers using concurrently
npm run dev

