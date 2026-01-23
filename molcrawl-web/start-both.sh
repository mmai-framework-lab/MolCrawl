#!/bin/bash

# MolCrawl Web 起動スクリプト
# バックエンドとフロントエンドの両方を起動します

set -e

# 環境変数チェック
if [ -z "$LEARNING_SOURCE_DIR" ]; then
  echo "⚠️  LEARNING_SOURCE_DIR が設定されていません"
  echo "例: export LEARNING_SOURCE_DIR='learning_source_20260106'"
  exit 1
fi

echo "=========================================="
echo "MolCrawl Web 起動"
echo "=========================================="
echo "LEARNING_SOURCE_DIR: $LEARNING_SOURCE_DIR"
echo ""

# 既存のプロセスを停止
echo "既存のプロセスを停止中..."
pkill -f "node server.js" 2>/dev/null || true
pkill -f "react-scripts start" 2>/dev/null || true
sleep 2

# バックエンドサーバーを起動
echo "バックエンドサーバーを起動中..."
LEARNING_SOURCE_DIR="$LEARNING_SOURCE_DIR" node server.js > /tmp/molcrawl-backend.log 2>&1 &
BACKEND_PID=$!
echo "  PID: $BACKEND_PID"

# バックエンドの起動を待つ
echo "バックエンドの起動を待機中..."
for i in {1..10}; do
  if lsof -i :3001 >/dev/null 2>&1; then
    echo "  ✅ バックエンドが起動しました (ポート 3001)"
    break
  fi
  if [ $i -eq 10 ]; then
    echo "  ❌ バックエンドの起動に失敗しました"
    cat /tmp/molcrawl-backend.log
    exit 1
  fi
  sleep 1
done

# フロントエンドを起動
echo "フロントエンドを起動中..."
npm start > /tmp/molcrawl-frontend.log 2>&1 &
FRONTEND_PID=$!
echo "  PID: $FRONTEND_PID"

echo ""
echo "=========================================="
echo "✅ 起動完了"
echo "=========================================="
echo "バックエンド: http://localhost:3001"
echo "フロントエンド: http://localhost:3000"
echo ""
echo "ログファイル:"
echo "  - バックエンド: /tmp/molcrawl-backend.log"
echo "  - フロントエンド: /tmp/molcrawl-frontend.log"
echo ""
echo "停止するには:"
echo "  pkill -f 'node server.js'"
echo "  pkill -f 'react-scripts start'"
echo "=========================================="
