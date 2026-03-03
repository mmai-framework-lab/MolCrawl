#!/bin/bash
# ==============================================================================
# Hugging Face Hub からモデルをダウンロードしてテストするスクリプト
# ==============================================================================
# 使用方法:
#   ./test_huggingface_download.sh <repo_id> [options]
#
# 例:
#   ./test_huggingface_download.sh deskull/rna-small-gpt2
#   ./test_huggingface_download.sh deskull/rna-small-gpt2 --test-generate --domain rna
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 使用方法を表示
show_usage() {
    cat << EOF
使用方法: $(basename "$0") <repo_id> [options]

引数:
  repo_id         Hugging Face Hub のリポジトリID (例: username/model-name)

オプション:
  --revision          ブランチ/タグ/コミット（デフォルト: main）
  --cache-dir         ダウンロード先のキャッシュディレクトリ
  --checkpoint-file   チェックポイントファイル名（デフォルト: ckpt.pt）
  --device            使用するデバイス (cpu, cuda, auto)
  --test-generate     テキスト生成テストを実行
  --domain            モデルのドメイン (rna, genome, protein_sequence, compounds, molecule_nat_lang)
  --max-tokens        生成する最大トークン数（デフォルト: 50）
  -v, --verbose       詳細な出力
  -h, --help          このヘルプを表示

例:
  # 基本的なテスト
  $(basename "$0") deskull/rna-small-gpt2

  # 生成テスト付き（RNAドメイン）
  $(basename "$0") deskull/rna-small-gpt2 --test-generate --domain rna

  # CPUで実行
  $(basename "$0") deskull/rna-small-gpt2 --device cpu
EOF
}

# 引数がない場合はヘルプを表示
if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

# 最初の引数をチェック
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

REPO_ID="$1"
shift

log_info "========================================"
log_info "Hugging Face Hub モデルテスト"
log_info "========================================"
log_info "リポジトリID: $REPO_ID"
log_info "========================================"

# Python依存関係の確認
log_info "依存関係を確認中..."

python3 -c "import torch" 2>/dev/null || {
    log_error "PyTorch がインストールされていません"
    log_info "インストール: pip install torch"
    exit 1
}

python3 -c "import huggingface_hub" 2>/dev/null || {
    log_error "huggingface_hub がインストールされていません"
    log_info "インストール: pip install huggingface_hub"
    exit 1
}

log_success "依存関係OK"

# Pythonスクリプトを実行
TEST_SCRIPT="$SCRIPT_DIR/test_huggingface_download.py"

if [ ! -f "$TEST_SCRIPT" ]; then
    log_error "テストスクリプトが見つかりません: $TEST_SCRIPT"
    exit 1
fi

log_info "テストを開始..."
python3 "$TEST_SCRIPT" "$REPO_ID" "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log_success "========================================"
    log_success "すべてのテストに合格しました！"
    log_success "========================================"
else
    log_error "========================================"
    log_error "一部のテストに失敗しました"
    log_error "========================================"
fi

exit $EXIT_CODE
