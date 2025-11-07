#!/bin/bash

#
# GPT-2 OMIM実データ評価パイプライン実行スクリプト
#
# このスクリプトは、実際のOMIMデータベースから取得したデータを使用して
# GPT-2モデルの遺伝性疾患予測性能を評価します。
# データ準備から評価、可視化までの全プロセスを自動で実行します。
#
# 注意: 
#   - このスクリプトはbootstraps/ディレクトリから実行されることを想定しています
#   - 実際のOMIMデータアクセスには有効な認証が必要です
#   - 設定ファイルに正しい認証付きURLが含まれている必要があります
#
# 使用方法:
#   ./run_gpt2_omim_real_evaluation.sh [options]
#

set -e  # エラー時に停止

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# スクリプトディレクトリ取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"  # プロジェクトルートディレクトリ

# LEARNING_SOURCE_DIRの確認
if [ -z "$LEARNING_SOURCE_DIR" ]; then
    echo "エラー: LEARNING_SOURCE_DIR環境変数が設定されていません"
    echo "実行前に以下を設定してください:"
    echo "  export LEARNING_SOURCE_DIR=/path/to/learning_source"
    exit 1
fi

# デフォルト設定
MODEL_SIZE="small"
BATCH_SIZE=16
TOKENIZER_PATH=""  # 空の場合は自動検出
# デフォルト出力先（-o/--output-dirで上書き可能）
OUTPUT_DIR="$LEARNING_SOURCE_DIR/genome_sequence/report/omim_real_evaluation"
DATA_DIR="$LEARNING_SOURCE_DIR/genome_sequence/data/omim_real"
DEFAULT_CONFIG="$PROJECT_ROOT/configs/omim_real_data.yaml"
EXISTING_OMIM_DIR=""  # 既存のOMIMデータディレクトリ
FORCE_DOWNLOAD=false
SKIP_DATA_PREP=false
SKIP_EVALUATION=false
SKIP_VISUALIZATION=false

# ヘルプメッセージ
show_usage() {
    cat << EOF
GPT-2 OMIM実データ評価パイプライン

使用方法: $0 [options]

オプション:
    --model_size SIZE           GPT-2モデルサイズ (small|medium|large, デフォルト: small)
    --tokenizer PATH            トークナイザーパス（指定しない場合は自動検出）
    --batch_size SIZE           バッチサイズ (デフォルト: 16)
    -o, --output_dir DIR        出力ディレクトリ (デフォルト: \$LEARNING_SOURCE_DIR/genome_sequence/report/omim_real_evaluation)
    --config FILE               実データ設定ファイル (デフォルト: configs/omim_real_data.yaml)
    --existing_omim_dir DIR     既存のOMIMデータディレクトリを指定（ダウンロード済みファイルの再利用）
    --force_download            データを強制再ダウンロード
    --skip_data_prep            データ準備をスキップ
    --skip_evaluation           評価をスキップ
    --skip_visualization        可視化をスキップ
    -h, --help                  このヘルプを表示

例:
    # 基本実行
    $0

    # カスタムモデルサイズとバッチサイズ
    $0 --model_size medium --batch_size 32

    # 既存のOMIMデータディレクトリを使用
    $0 --existing_omim_dir /path/to/existing/omim/data

    # 強制再ダウンロード付き実行
    $0 --force_download

    # カスタム設定ファイル使用
    $0 --config /path/to/custom/config.yaml

注意事項:
    - 実際のOMIMデータアクセスには有効な認証が必要です
    - 設定ファイルに正しい認証付きURLが含まれている必要があります
    - データは\$LEARNING_SOURCE_DIR配下に保存されます
EOF
}

# パラメータ解析
CONFIG_FILE="$DEFAULT_CONFIG"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --existing_omim_dir)
            EXISTING_OMIM_DIR="$2"
            shift 2
            ;;
        --force_download)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --skip_data_prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --skip_evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --skip_visualization)
            SKIP_VISUALIZATION=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 設定表示
echo "=== GPT-2 OMIM実データ評価パイプライン開始 ==="
echo "モデルサイズ: $MODEL_SIZE"
echo "バッチサイズ: $BATCH_SIZE"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "データディレクトリ: $DATA_DIR"
echo "設定ファイル: $CONFIG_FILE"
if [[ -n "$EXISTING_OMIM_DIR" ]]; then
    echo "既存OMIMディレクトリ: $EXISTING_OMIM_DIR"
fi
echo "強制ダウンロード: $FORCE_DOWNLOAD"
echo ""

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

# パス設定
MODEL_PATH="$PROJECT_ROOT/gpt2-output/genome_sequence-$MODEL_SIZE/ckpt.pt"
DATA_PATH="$DATA_DIR/omim_real_evaluation_dataset.csv"

# 設定ファイル確認
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "エラー: 設定ファイルが見つかりません: $CONFIG_FILE"
    exit 1
fi

# モデルファイル存在チェック（評価を実行する場合のみ）
if [ "$SKIP_EVALUATION" = false ]; then
    if [ ! -f "$MODEL_PATH" ]; then
        echo "エラー: GPT-2モデルファイルが見つかりません: $MODEL_PATH"
        echo "利用可能なモデル:"
        ls -la "$PROJECT_ROOT/gpt2-output/" 2>/dev/null || echo "  gpt2-outputディレクトリが存在しません"
        exit 1
    fi
fi

# Python環境チェック
if ! command -v python &> /dev/null; then
    echo "エラー: Pythonが見つかりません"
    exit 1
fi

# 必要なPythonパッケージチェック（GPT-2評価用）
echo "Pythonパッケージを確認中..."
python -c "
import sys
required_packages = ['pandas', 'numpy', 'torch', 'sentencepiece', 'sklearn', 'matplotlib', 'seaborn', 'yaml', 'requests']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'エラー: 必要なパッケージが見つかりません: {missing_packages}')
    print('以下のコマンドでインストールしてください: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('全ての必要なパッケージが見つかりました。')
"

if [[ $? -ne 0 ]]; then
    echo "エラー: 必要なPythonパッケージが不足しています"
    exit 1
fi

echo "GPT-2環境変数設定: LEARNING_SOURCE_DIR=$LEARNING_SOURCE_DIR"
echo ""

# =============================================================================
# データ準備フェーズ
# =============================================================================

if [ "$SKIP_DATA_PREP" = false ]; then
    echo "=== データ準備フェーズ ==="
    
    # 既存のOMIMディレクトリが指定されている場合
    if [[ -n "$EXISTING_OMIM_DIR" ]]; then
        echo "既存のOMIMデータディレクトリを使用します: $EXISTING_OMIM_DIR"
        
        # 既存ディレクトリの検証
        if [[ ! -d "$EXISTING_OMIM_DIR" ]]; then
            echo "エラー: 指定されたOMIMディレクトリが存在しません: $EXISTING_OMIM_DIR"
            exit 1
        fi
        
        # 必要なOMIMファイルの確認
        REQUIRED_FILES=("mim2gene.txt" "mimTitles.txt" "genemap2.txt" "morbidmap.txt")
        MISSING_FILES=()
        
        for file in "${REQUIRED_FILES[@]}"; do
            if [[ ! -f "$EXISTING_OMIM_DIR/$file" ]]; then
                MISSING_FILES+=("$file")
            fi
        done
        
        if [ ${#MISSING_FILES[@]} -gt 0 ]; then
            echo "警告: 以下のファイルが見つかりません:"
            for file in "${MISSING_FILES[@]}"; do
                echo "  - $file"
            done
            echo "不完全なデータで処理を続行します..."
        else
            echo "✓ 全ての必要なOMIMファイルが見つかりました"
        fi
        
        # 既存ファイルをデータディレクトリにコピーまたはシンボリックリンク
        mkdir -p "$DATA_DIR"
        
        echo "既存のOMIMファイルをリンク中..."
        for file in "${REQUIRED_FILES[@]}"; do
            if [[ -f "$EXISTING_OMIM_DIR/$file" ]]; then
                # シンボリックリンクを作成（既に存在する場合は削除）
                rm -f "$DATA_DIR/$file"
                ln -s "$EXISTING_OMIM_DIR/$file" "$DATA_DIR/$file"
                echo "  ✓ $file"
            fi
        done
        
        echo "既存データの準備完了"
    else
        echo "OMIM実データをダウンロード・準備中..."
    fi
    
    cd "$PROJECT_ROOT"
    
    # Pythonコマンド引数を準備
    DATA_PREP_ARGS=(
        "scripts/evaluation/gpt2/omim_data_preparation.py"
        "--mode" "real"
        "--output_dir" "$DATA_DIR"
        "--config" "$CONFIG_FILE"
    )
    
    # 既存ディレクトリが指定されている場合は追加
    if [[ -n "$EXISTING_OMIM_DIR" ]]; then
        DATA_PREP_ARGS+=("--existing_omim_dir" "$EXISTING_OMIM_DIR")
    fi
    
    if [[ "$FORCE_DOWNLOAD" == "true" ]]; then
        DATA_PREP_ARGS+=("--force_download")
        echo "📥 データを強制再ダウンロード中..."
    fi
    
    python "${DATA_PREP_ARGS[@]}"
    
    if [[ $? -ne 0 ]]; then
        echo "エラー: 実データ準備に失敗しました"
        exit 1
    fi
    
    echo "データ準備完了"
    echo ""
fi

# データファイル存在チェック
if [ ! -f "$DATA_PATH" ]; then
    echo "エラー: データファイルが見つかりません: $DATA_PATH"
    echo "先にデータ準備を実行してください（--skip_data_prepを外す）"
    exit 1
fi

# =============================================================================
# モデル評価フェーズ
# =============================================================================

if [ "$SKIP_EVALUATION" = false ]; then
    echo "=== モデル評価フェーズ ==="
    echo "GPT-2 OMIM実データ評価を実行中..."
    echo "モデル: $MODEL_PATH"
    echo "データ: $DATA_PATH"
    
    cd "$PROJECT_ROOT"
    
    # Pythonコマンド引数を準備
    EVAL_ARGS=(
        "scripts/evaluation/gpt2/omim_evaluation.py"
        --model_path "$MODEL_PATH"
        --data_path "$DATA_PATH"
        --output_dir "$OUTPUT_DIR"
        --batch_size "$BATCH_SIZE"
    )
    
    # トークナイザーパスが指定されている場合は追加
    if [[ -n "$TOKENIZER_PATH" ]]; then
        EVAL_ARGS+=(--tokenizer_path "$TOKENIZER_PATH")
        log_info "トークナイザー: $TOKENIZER_PATH"
    else
        log_info "トークナイザー: 自動検出"
    fi
    
    python "${EVAL_ARGS[@]}"
    
    if [[ $? -ne 0 ]]; then
        echo "エラー: モデル評価に失敗しました"
        exit 1
    fi
    
    echo "モデル評価完了"
    echo ""
fi

# =============================================================================
# 可視化フェーズ
# =============================================================================

if [ "$SKIP_VISUALIZATION" = false ]; then
    echo "=== 可視化フェーズ ==="
    echo "評価結果の可視化を実行中..."
    
    python "$PROJECT_ROOT/scripts/evaluation/gpt2/omim_visualization.py" \
        --results_dir "$OUTPUT_DIR"
    
    if [[ $? -ne 0 ]]; then
        echo "エラー: 可視化に失敗しました"
        exit 1
    fi
    
    echo "可視化完了"
    echo ""
fi

# =============================================================================
# 完了メッセージ
# =============================================================================

echo "=== GPT-2 OMIM実データ評価パイプライン完了 ==="
echo ""

# 結果サマリー表示
RESULTS_FILE="$OUTPUT_DIR/omim_evaluation_results.json"
if [[ -f "$RESULTS_FILE" ]]; then
    echo "📋 評価メトリクス:"
    echo "======================"
    
    # JSON結果から主要メトリクスを抽出して表示
    python -c "
import json
import sys

try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    
    metrics = results.get('overall_metrics', results)
    print(f'  • Accuracy:    {metrics.get(\"accuracy\", 0):.4f}')
    print(f'  • Precision:   {metrics.get(\"precision\", 0):.4f}')
    print(f'  • Recall:      {metrics.get(\"recall\", 0):.4f}')
    print(f'  • F1-score:    {metrics.get(\"f1_score\", 0):.4f}')
    print(f'  • ROC-AUC:     {metrics.get(\"roc_auc\", 0):.4f}')
    print(f'  • PR-AUC:      {metrics.get(\"pr_auc\", 0):.4f}')
    print(f'  • Sensitivity: {metrics.get(\"sensitivity\", 0):.4f}')
    print(f'  • Specificity: {metrics.get(\"specificity\", 0):.4f}')
except Exception as e:
    print(f'  メトリクスの読み込みに失敗しました: {e}', file=sys.stderr)
    sys.exit(0)
" 2>/dev/null || echo "  メトリクスを読み込めませんでした"
    echo ""
fi

echo "📁 出力ディレクトリ: $OUTPUT_DIR"

if [ "$SKIP_DATA_PREP" = false ]; then
    echo "📊 データ: $DATA_DIR"
fi

echo ""
echo "=== 出力ファイル ==="
echo "評価結果: $OUTPUT_DIR/omim_evaluation_results.json"
echo "詳細レポート: $OUTPUT_DIR/omim_evaluation_report.txt"
echo "可視化結果: $OUTPUT_DIR/visualizations/"
echo "HTMLレポート: $OUTPUT_DIR/visualizations/omim_evaluation_report.html"

echo ""
echo "=== OMIMデータベースについて ==="
cat << 'EOF'
OMIM (Online Mendelian Inheritance in Man) は遺伝性疾患の包括的データベースです。
- 25,000以上の遺伝性疾患・遺伝子情報を収録
- 単一遺伝子疾患、複合遺伝、染色体異常の詳細な分類
- 遺伝形式(常染色体優性/劣性、X連鎖、ミトコンドリア)の明確な分類
- 実際のOMIMデータ使用には登録が必要: https://omim.org/
- このスクリプトは実際のOMIMデータベースからデータを取得します
EOF

echo ""
echo "✅ 全ての処理が正常に完了しました"
