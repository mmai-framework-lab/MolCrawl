#!/bin/bash

# RIKEN Dataset Management Interface Startup Script
# This script starts both the API server and React frontend

set -e

echo "🧬 Starting RIKEN Dataset Management Interface..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js first.${NC}"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed. Please install npm first.${NC}"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
API_DIR="$SCRIPT_DIR/api"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo -e "${BLUE}📁 Script directory: $SCRIPT_DIR${NC}"

# Function to install dependencies
install_dependencies() {
    local dir=$1
    local name=$2

    echo -e "${YELLOW}📦 Installing $name dependencies...${NC}"
    cd "$dir"

    if [ ! -f "package.json" ]; then
        echo -e "${RED}❌ package.json not found in $dir${NC}"
        return 1
    fi

    if [ ! -d "node_modules" ] || [ ! -f "package-lock.json" ]; then
        echo -e "${BLUE}🔄 Running npm install...${NC}"
        npm install
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ $name dependencies installed successfully${NC}"
        else
            echo -e "${RED}❌ Failed to install $name dependencies${NC}"
            return 1
        fi
    else
        echo -e "${GREEN}✅ $name dependencies already installed${NC}"
    fi
}

# Install API dependencies
if [ -d "$API_DIR" ]; then
    install_dependencies "$API_DIR" "API server"
else
    echo -e "${RED}❌ API directory not found: $API_DIR${NC}"
    exit 1
fi

# Install Frontend dependencies
if [ -d "$FRONTEND_DIR" ]; then
    install_dependencies "$FRONTEND_DIR" "Frontend"
else
    echo -e "${RED}❌ Frontend directory not found: $FRONTEND_DIR${NC}"
    exit 1
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 1
    else
        return 0
    fi
}

# Check if required ports are available
API_PORT=3001
FRONTEND_PORT=3000

if ! check_port $API_PORT; then
    echo -e "${YELLOW}⚠️  Port $API_PORT is already in use. Trying to find the process...${NC}"
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

if ! check_port $FRONTEND_PORT; then
    echo -e "${YELLOW}⚠️  Port $FRONTEND_PORT is already in use. Trying to find the process...${NC}"
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Create log directory
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}📊 Starting services...${NC}"

# Start API server in background
cd "$API_DIR"
echo -e "${YELLOW}🚀 Starting API server on port $API_PORT...${NC}"
nohup npm start > "$LOG_DIR/api.log" 2>&1 &
API_PID=$!
echo $API_PID > "$LOG_DIR/api.pid"
echo -e "${GREEN}✅ API server started (PID: $API_PID)${NC}"

# Wait a moment for API server to start
sleep 3

# Check if API server is running
if ! ps -p $API_PID > /dev/null; then
    echo -e "${RED}❌ API server failed to start. Check logs: $LOG_DIR/api.log${NC}"
    cat "$LOG_DIR/api.log"
    exit 1
fi

# Test API server
echo -e "${BLUE}🔍 Testing API server...${NC}"
if curl -s "http://localhost:$API_PORT/health" > /dev/null; then
    echo -e "${GREEN}✅ API server is responding${NC}"
else
    echo -e "${YELLOW}⚠️  API server may not be fully ready yet${NC}"
fi

# Start Frontend server in background
cd "$FRONTEND_DIR"
echo -e "${YELLOW}🎨 Starting Frontend server on port $FRONTEND_PORT...${NC}"
nohup npm start > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$LOG_DIR/frontend.pid"
echo -e "${GREEN}✅ Frontend server started (PID: $FRONTEND_PID)${NC}"

# Store both PIDs for cleanup
echo "$API_PID,$FRONTEND_PID" > "$LOG_DIR/services.pid"

echo ""
echo -e "${GREEN}🎉 RIKEN Dataset Management Interface is starting up!${NC}"
echo "=================================================="
echo -e "${BLUE}📊 API Server:      http://localhost:$API_PORT${NC}"
echo -e "${BLUE}🎨 Frontend:        http://localhost:$FRONTEND_PORT${NC}"
echo -e "${BLUE}📝 API Logs:        $LOG_DIR/api.log${NC}"
echo -e "${BLUE}📝 Frontend Logs:   $LOG_DIR/frontend.log${NC}"
echo ""
echo -e "${YELLOW}💡 The frontend will automatically open in your browser.${NC}"
echo -e "${YELLOW}💡 Press Ctrl+C to stop all services.${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down services...${NC}"

    if [ -f "$LOG_DIR/services.pid" ]; then
        PIDS=$(cat "$LOG_DIR/services.pid")
        IFS=',' read -ra PID_ARRAY <<< "$PIDS"
        for pid in "${PID_ARRAY[@]}"; do
            if ps -p $pid > /dev/null; then
                echo -e "${BLUE}🔄 Stopping process $pid...${NC}"
                kill $pid 2>/dev/null || true
            fi
        done
        rm -f "$LOG_DIR/services.pid"
    fi

    # Clean up individual PID files
    if [ -f "$LOG_DIR/api.pid" ]; then
        API_PID=$(cat "$LOG_DIR/api.pid")
        kill $API_PID 2>/dev/null || true
        rm -f "$LOG_DIR/api.pid"
    fi

    if [ -f "$LOG_DIR/frontend.pid" ]; then
        FRONTEND_PID=$(cat "$LOG_DIR/frontend.pid")
        kill $FRONTEND_PID 2>/dev/null || true
        rm -f "$LOG_DIR/frontend.pid"
    fi

    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for user to stop the services
echo -e "${BLUE}⏳ Services are running. Press Ctrl+C to stop...${NC}"

# Monitor the services
while true; do
    sleep 5

    # Check if API server is still running
    if ! ps -p $API_PID > /dev/null; then
        echo -e "${RED}❌ API server stopped unexpectedly${NC}"
        break
    fi

    # Check if Frontend server is still running
    if ! ps -p $FRONTEND_PID > /dev/null; then
        echo -e "${RED}❌ Frontend server stopped unexpectedly${NC}"
        break
    fi
done

cleanup
