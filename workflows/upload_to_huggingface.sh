#!/bin/bash
# ==============================================================================
# Hugging Face Hub へモデルをアップロードするスクリプト
# ==============================================================================
# 使用方法:
#   ./upload_to_huggingface.sh <model_path> <repo_id> [options]
#
# 例:
#   ./upload_to_huggingface.sh ../gpt2-output/rna-small matsubara-riken/rna-small-gpt2
#   ./upload_to_huggingface.sh ../gpt2-output/rna-small matsubara-riken/rna-small-gpt2 --private
#
# 事前準備:
#   1. Hugging Face アカウントを作成
#   2. アクセストークンを取得: https://huggingface.co/settings/tokens
#   3. 認証: huggingface-cli login
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
使用方法: $(basename "$0") <model_path> <repo_id> [options]

引数:
  model_path    アップロードするモデルのパス（ディレクトリまたはチェックポイントファイル）
  repo_id       Hugging Face Hub のリポジトリID (例: username/model-name)

オプション:
  --private           プライベートリポジトリとして作成
  --commit-message    コミットメッセージ（デフォルト: "Upload model"）
  --model-type        モデルタイプ (gpt2, bert, dnabert2, esm2, rnaformer, chemberta2)
  --tokenizer-path    トークナイザーのパス（省略時はmodel_pathから自動検出）
  --config-path       設定ファイルのパス（省略時はmodel_pathから自動検出）
  --create-model-card モデルカードを自動生成
  --dry-run           実際にはアップロードせず、何が行われるか表示
  -h, --help          このヘルプを表示

環境変数:
  HF_TOKEN            Hugging Face APIトークン（未設定の場合は huggingface-cli login が必要）

例:
  # 基本的な使用
  $(basename "$0") ../gpt2-output/rna-small matsubara-riken/rna-small-gpt2

  # プライベートリポジトリとして、モデルカード付きでアップロード
  $(basename "$0") ../gpt2-output/rna-small matsubara-riken/rna-small-gpt2 \\
      --private --create-model-card --model-type gpt2

  # ドライラン（確認のみ）
  $(basename "$0") ../gpt2-output/rna-small matsubara-riken/rna-small-gpt2 --dry-run
EOF
}

# デフォルト値
PRIVATE=false
COMMIT_MESSAGE="Upload model"
MODEL_TYPE=""
TOKENIZER_PATH=""
CONFIG_PATH=""
CREATE_MODEL_CARD=false
DRY_RUN=false

# 引数解析
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

MODEL_PATH="$1"
REPO_ID="$2"
shift 2

while [[ $# -gt 0 ]]; do
    case $1 in
        --private)
            PRIVATE=true
            shift
            ;;
        --commit-message)
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --create-model-card)
            CREATE_MODEL_CARD=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "不明なオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

# モデルパスを絶対パスに変換
if [[ ! "$MODEL_PATH" = /* ]]; then
    MODEL_PATH="$(cd "$SCRIPT_DIR" && cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
fi

# モデルパスの存在確認
if [ ! -e "$MODEL_PATH" ]; then
    log_error "モデルパスが見つかりません: $MODEL_PATH"
    exit 1
fi

log_info "========================================"
log_info "Hugging Face Hub アップロード"
log_info "========================================"
log_info "モデルパス: $MODEL_PATH"
log_info "リポジトリID: $REPO_ID"
log_info "プライベート: $PRIVATE"
log_info "コミットメッセージ: $COMMIT_MESSAGE"
[ -n "$MODEL_TYPE" ] && log_info "モデルタイプ: $MODEL_TYPE"
[ -n "$TOKENIZER_PATH" ] && log_info "トークナイザーパス: $TOKENIZER_PATH"
[ "$CREATE_MODEL_CARD" = true ] && log_info "モデルカード: 自動生成"
[ "$DRY_RUN" = true ] && log_warning "ドライランモード（実際のアップロードは行いません）"
log_info "========================================"

# Python依存関係の確認
log_info "依存関係を確認中..."

python3 -c "import huggingface_hub" 2>/dev/null || {
    log_error "huggingface_hub がインストールされていません"
    log_info "インストール: pip install huggingface_hub"
    exit 1
}

# 認証確認（ドライランの場合はスキップ）
if [ "$DRY_RUN" = false ]; then
    log_info "Hugging Face 認証を確認中..."
    if [ -z "$HF_TOKEN" ]; then
        if ! python3 -c "from huggingface_hub import HfFolder; token = HfFolder.get_token(); exit(0 if token else 1)" 2>/dev/null; then
            log_error "Hugging Face にログインしていません"
            log_info "以下のいずれかを実行してください:"
            log_info "  1. huggingface-cli login"
            log_info "  2. export HF_TOKEN='your_token_here'"
            exit 1
        fi
        log_success "認証済み（ローカルトークンを使用）"
    else
        log_success "認証済み（HF_TOKEN環境変数を使用）"
    fi
else
    log_info "ドライランのため認証チェックをスキップ"
fi

# Pythonスクリプトを実行
UPLOAD_SCRIPT="$SCRIPT_DIR/upload_to_huggingface.py"

if [ ! -f "$UPLOAD_SCRIPT" ]; then
    log_error "アップロードスクリプトが見つかりません: $UPLOAD_SCRIPT"
    exit 1
fi

# Python引数を構築
PYTHON_ARGS=(
    "$UPLOAD_SCRIPT"
    "$MODEL_PATH"
    "$REPO_ID"
    --commit-message "$COMMIT_MESSAGE"
)

[ "$PRIVATE" = true ] && PYTHON_ARGS+=(--private)
[ -n "$MODEL_TYPE" ] && PYTHON_ARGS+=(--model-type "$MODEL_TYPE")
[ -n "$TOKENIZER_PATH" ] && PYTHON_ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
[ -n "$CONFIG_PATH" ] && PYTHON_ARGS+=(--config-path "$CONFIG_PATH")
[ "$CREATE_MODEL_CARD" = true ] && PYTHON_ARGS+=(--create-model-card)
[ "$DRY_RUN" = true ] && PYTHON_ARGS+=(--dry-run)

log_info "アップロードを開始..."
python3 "${PYTHON_ARGS[@]}"

if [ $? -eq 0 ]; then
    log_success "========================================"
    log_success "アップロード完了！"
    log_success "URL: https://huggingface.co/$REPO_ID"
    log_success "========================================"
else
    log_error "アップロードに失敗しました"
    exit 1
fi
