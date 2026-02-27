#!/bin/bash

# 実験管理システムの簡易デモ
# このスクリプトはシステムの主要機能を実演します

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================"
echo "🧪 実験管理システム - デモンストレーション"
echo "================================================"
echo ""

# PYTHONPATHを設定
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "Step 1: システムのテスト"
echo "----------------------------------------"
 test_experiment_system.py
echo ""

echo "Step 2: サンプル実験の実行"
echo "----------------------------------------"

echo "📊 実験1: データ準備（コンテキストマネージャー）"
 examples/experiment_tracking_example.py --example context
echo ""

echo "🧠 実験2: 評価（手動トラッキング）"
 examples/experiment_tracking_example.py --example manual
echo ""

echo "❌ 実験3: 失敗シナリオ"
 examples/experiment_tracking_example.py --example failure
echo ""

echo "Step 3: 実験一覧の表示"
echo "----------------------------------------"
 examples/experiment_tracking_example.py --example list
echo ""

echo "================================================"
echo "✅ デモンストレーション完了"
echo "================================================"
echo ""
echo "次のステップ:"
echo ""
echo "1. システムを起動:"
echo "   ./start_experiment_system.sh"
echo ""
echo "2. Webブラウザでアクセス:"
echo "   http://localhost:3000"
echo ""
echo "3. Experimentsタブをクリックして実験を確認"
echo ""
echo "4. 詳細ドキュメント:"
echo "   - EXPERIMENT_TRACKING_QUICKSTART.md"
echo "   - EXPERIMENT_TRACKING_README.md"
echo "   - EXPERIMENT_TRACKING_ARCHITECTURE.md"
echo ""
echo "================================================"
