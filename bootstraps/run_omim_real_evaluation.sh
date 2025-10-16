#!/bin/bash

# ====================================================================
# OMIM Real Data Evaluation Script
# ====================================================================
# 実際のOMIMデータを使用してゲノム配列モデルの遺伝性疾患予測性能を評価
# ====================================================================

set -e  # エラーで停止

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
PROJECT_ROOT="$SCRIPT_DIR"

# デフォルト設定
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/outputs/omim_real_evaluation"
DEFAULT_CONFIG="$PROJECT_ROOT/configs/omim_real_data.yaml"
DEFAULT_MODEL_DIR="$PROJECT_ROOT/fundamental_models_202407"
FORCE_DOWNLOAD=false

# 使用方法表示
show_usage() {
    cat << EOF
OMIM Real Data Evaluation Script

使用方法:
    $0 [OPTIONS]

オプション:
    -o, --output-dir DIR    出力ディレクトリ (デフォルト: $DEFAULT_OUTPUT_DIR)
    -c, --config FILE       実データ設定ファイル (デフォルト: $DEFAULT_CONFIG)
    -m, --model-dir DIR     モデルディレクトリ (デフォルト: $DEFAULT_MODEL_DIR)
    -f, --force-download    データを強制再ダウンロード
    -h, --help              このヘルプを表示

実行例:
    # 基本実行
    $0

    # カスタム出力ディレクトリ指定
    $0 --output-dir /path/to/custom/output

    # 強制再ダウンロード付き実行
    $0 --force-download

    # カスタム設定ファイル使用
    $0 --config /path/to/custom/config.yaml

注意事項:
    - 実際のOMIMデータアクセスには有効な認証が必要です
    - 設定ファイルに正しい認証付きURLが含まれている必要があります
    - データはリポジトリには保存されません
EOF
}

# パラメータ解析
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
CONFIG_FILE="$DEFAULT_CONFIG"
MODEL_DIR="$DEFAULT_MODEL_DIR"

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -f|--force-download)
            FORCE_DOWNLOAD=true
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

# 評価実行関数
run_omim_real_evaluation() {
    log_info "=== OMIM Real Data Evaluation Start ==="
    log_info "出力ディレクトリ: $OUTPUT_DIR"
    log_info "設定ファイル: $CONFIG_FILE"
    log_info "モデルディレクトリ: $MODEL_DIR"
    log_info "強制ダウンロード: $FORCE_DOWNLOAD"

    # ディレクトリ作成
    mkdir -p "$OUTPUT_DIR"

    # 設定ファイル確認
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "設定ファイルが見つかりません: $CONFIG_FILE"
        exit 1
    fi

    # 1. 実データ準備
    log_info "=== Step 1: OMIM Real Data Preparation ==="
    local data_prep_args="--mode real --output_dir $OUTPUT_DIR --config $CONFIG_FILE"
    if [[ "$FORCE_DOWNLOAD" == "true" ]]; then
        data_prep_args="$data_prep_args --force_download"
    fi

    if python3 "$PROJECT_ROOT/scripts/omim_data_preparation.py" $data_prep_args; then
        log_success "実データ準備完了"
    else
        log_error "実データ準備に失敗"
        exit 1
    fi

    # 2. データファイル確認
    local dataset_file="$OUTPUT_DIR/data/omim_real_evaluation_dataset.csv"
    if [[ ! -f "$dataset_file" ]]; then
        log_error "データセットファイルが見つかりません: $dataset_file"
        exit 1
    fi

    # 3. モデル評価実行
    log_info "=== Step 2: Model Evaluation ==="
    
    # 環境変数設定
    export LEARNING_SOURCE_DIR="learning_source_202508"
    
    local eval_args=""
    eval_args="$eval_args --data_path $dataset_file"
    eval_args="$eval_args --output_dir $OUTPUT_DIR"
    eval_args="$eval_args --model_path dummy_model"

    if python3 "$PROJECT_ROOT/scripts/omim_evaluation.py" $eval_args; then
        log_success "モデル評価完了"
    else
        log_error "モデル評価に失敗"
        exit 1
    fi

    # 4. 結果可視化
    log_info "=== Step 3: Results Visualization ==="
    local viz_args=""
    viz_args="$viz_args --results_dir $OUTPUT_DIR"
    viz_args="$viz_args --output_dir $OUTPUT_DIR"

    if python3 "$PROJECT_ROOT/scripts/omim_visualization.py" $viz_args; then
        log_success "結果可視化完了"
    else
        log_error "結果可視化に失敗"
        exit 1
    fi

    # 5. 結果サマリー表示
    log_info "=== Step 4: Results Summary ==="
    local results_file="$OUTPUT_DIR/omim_evaluation_results.json"
    if [[ -f "$results_file" ]]; then
        log_info "評価結果ファイル: $results_file"
        
        # JSON結果から主要メトリクスを抽出して表示
        if command -v jq >/dev/null 2>&1; then
            local accuracy=$(jq -r '.overall_metrics.accuracy // "N/A"' "$results_file")
            local f1_score=$(jq -r '.overall_metrics.f1_score // "N/A"' "$results_file")
            local precision=$(jq -r '.overall_metrics.precision // "N/A"' "$results_file")
            local recall=$(jq -r '.overall_metrics.recall // "N/A"' "$results_file")
            
            log_success "=== 評価メトリクス ==="
            log_success "Accuracy: $accuracy"
            log_success "F1 Score: $f1_score"
            log_success "Precision: $precision"
            log_success "Recall: $recall"
        else
            log_warn "jqコマンドが見つかりません。詳細結果は $results_file を確認してください"
        fi
    fi

    # 6. 出力ファイル一覧
    log_info "=== Step 5: Output Files ==="
    log_info "生成されたファイル:"
    find "$OUTPUT_DIR" -type f -name "*.csv" -o -name "*.json" -o -name "*.png" -o -name "*.html" | sort | while read -r file; do
        local rel_path="${file#$OUTPUT_DIR/}"
        log_info "  - $rel_path"
    done

    log_success "=== OMIM Real Data Evaluation Completed Successfully ==="
    log_info "結果ディレクトリ: $OUTPUT_DIR"
}

# メイン実行
main() {
    # Python環境確認
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "Python 3 が見つかりません"
        exit 1
    fi

    # 必要な Python パッケージ確認
    log_info "Python環境確認中..."
    python3 -c "
import sys
required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'yaml', 'requests']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'以下のパッケージが不足しています: {missing_packages}')
    print('pip install でインストールしてください')
    sys.exit(1)
else:
    print('必要なパッケージは揃っています')
"

    if [[ $? -ne 0 ]]; then
        log_error "Python環境に問題があります"
        exit 1
    fi

    # 評価実行
    run_omim_real_evaluation
}

# トラップ設定（Ctrl+C対応）
trap 'log_warn "処理が中断されました"; exit 130' INT

# スクリプト実行
main "$@"
