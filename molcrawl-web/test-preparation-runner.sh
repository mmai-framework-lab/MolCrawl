#!/bin/bash

# 準備スクリプト実行機能のテストスクリプト

BASE_URL="http://localhost:3001"

echo "=========================================="
echo "準備スクリプトランナー API テスト"
echo "=========================================="
echo ""

# 1. スクリプト一覧の取得
echo "1️⃣ 利用可能なスクリプト一覧を取得..."
curl -s "${BASE_URL}/api/preparation-runner/scripts" | jq '.' || echo "エラー: スクリプト一覧の取得に失敗"
echo ""
echo ""

# 2. すべての実行状態を確認
echo "2️⃣ 現在の実行状態を確認..."
curl -s "${BASE_URL}/api/preparation-runner/all-status" | jq '.' || echo "エラー: 状態取得に失敗"
echo ""
echo ""

# 3. protein_sequenceのphase01の状態を確認
echo "3️⃣ protein_sequence Phase01の状態を確認..."
curl -s "${BASE_URL}/api/preparation-runner/status/protein_sequence/phase01" | jq '.' || echo "エラー: 状態取得に失敗"
echo ""
echo ""

echo "=========================================="
echo "テスト完了"
echo "=========================================="
echo ""
echo "⚠️  注意: 実際にスクリプトを実行するには以下のコマンドを使用してください"
echo ""
echo "curl -X POST -H \"Content-Type: application/json\" \\"
echo "  -d '{\"dataset\": \"protein_sequence\", \"phase\": \"phase01\"}' \\"
echo "  ${BASE_URL}/api/preparation-runner/start"
echo ""
