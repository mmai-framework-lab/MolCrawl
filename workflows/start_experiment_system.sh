#!/bin/bash

# 実験管理システム起動スクリプト

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================"
echo "🧪 MolCrawl Experiment Management System"
echo "================================================"

# 1. APIサーバーを起動
echo ""
echo "Starting API Server..."
 start_api_server.py &
API_PID=$!
echo "API Server PID: $API_PID"

# 少し待機
sleep 3

# 2. Webフロントエンドを起動
echo ""
echo "Starting Web Frontend..."
cd molcrawl-web
npm run dev &
WEB_PID=$!
echo "Web Frontend PID: $WEB_PID"

echo ""
echo "================================================"
echo "✓ System Started"
echo "================================================"
echo "API Server:  http://localhost:8000"
echo "API Docs:    http://localhost:8000/docs"
echo "Web UI:      http://localhost:3000"
echo "Experiments: http://localhost:3000/experiments"
echo "================================================"
echo ""
echo "Press Ctrl+C to stop all services"

# Ctrl+Cでクリーンアップ
trap "echo 'Stopping services...'; kill $API_PID $WEB_PID 2>/dev/null; exit" INT

# 待機
wait
