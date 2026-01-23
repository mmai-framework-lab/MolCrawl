#!/bin/bash

# 実験管理システムのセットアップスクリプト
# このスクリプトは初回セットアップ時に1回だけ実行します

set -e  # エラーが発生したら即座に終了

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================="
echo "実験管理システム セットアップ"
echo "=================================="
echo ""

# 1. 必要なディレクトリを作成
echo "📁 ディレクトリを作成中..."
mkdir -p experiment_data
mkdir -p logs
mkdir -p examples

# 2. Pythonパスの設定
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
echo "✅ PYTHONPATH設定: ${PYTHONPATH}"

# 3. Condaパスの確認
if [ -f "${SCRIPT_DIR}/miniconda/bin/activate" ]; then
    echo "✅ Conda環境を検出しました"
    source "${SCRIPT_DIR}/miniconda/bin/activate" conda
else
    echo "⚠️  Conda環境が見つかりません。Pythonの仮想環境が必要です。"
fi

# 4. Pythonパッケージの確認
echo ""
echo "📦 Pythonパッケージを確認中..."
python -c "import fastapi, uvicorn, sqlite3" 2>/dev/null && echo "✅ 必要なパッケージがインストールされています" || {
    echo "❌ 必要なパッケージが不足しています"
    echo "以下のコマンドでインストールしてください:"
    echo "  pip install fastapi uvicorn"
    exit 1
}

# 5. データベースの初期化テスト
echo ""
echo "🗄️  データベースを初期化中..."
python -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from src.experiment_tracker import ExperimentTracker
tracker = ExperimentTracker()
print('✅ データベースが正常に初期化されました')
print(f'📍 データベース位置: {tracker.db.db_path}')
"

# 6. サンプル実験の作成
echo ""
echo "🔬 サンプル実験を作成中..."
python examples/experiment_tracking_example.py

# 7. Node.js パッケージの確認（molcrawl-web用）
echo ""
echo "📦 Node.js パッケージを確認中..."
if [ -d "molcrawl-web/node_modules" ]; then
    echo "✅ Node.js パッケージがインストールされています"
else
    echo "⚠️  Node.js パッケージがインストールされていません"
    echo "以下のコマンドでインストールしてください:"
    echo "  cd molcrawl-web && npm install"
fi

# 8. 完了メッセージ
echo ""
echo "=================================="
echo "✅ セットアップ完了！"
echo "=================================="
echo ""
echo "次のコマンドでシステムを起動できます:"
echo "  ./start_experiment_system.sh"
echo ""
echo "または、個別に起動する場合:"
echo "  1. APIサーバー: python start_api_server.py"
echo "  2. Webフロントエンド: cd molcrawl-web && npm run dev"
echo ""
