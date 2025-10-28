#!/bin/bash
#"""
#ClinVar評価パイプライン実行スクリプト
#
#このスクリプトは、ClinVarデータの取得から評価、可視化までの
#全プロセスを自動で実行します。
#"""

set -e  # エラー時に停止

# 設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$LEARNING_SOURCE_DIR/$PROJECT_ROOT/clinvar_evaluation_results"
DATA_DIR="$OUTPUT_DIR/data"
MODELS_DIR="$PROJECT_ROOT/$LEARNING_SOURCE_DIR/genome_sequence/gpt2-output"

# デフォルト設定
MODEL_SIZE="small"
SEQUENCE_LENGTH=100
MAX_SAMPLES=1000
BATCH_SIZE=16

# ヘルプ表示
show_help() {
    cat << EOF
ClinVar評価パイプライン

使用法: $0 [オプション]

オプション:
    -m, --model-size SIZE       モデルサイズ (small/medium/large/xl) [default: small]
    -s, --sequence-length LEN   配列長 [default: 100]
    -n, --max-samples NUM       クラスあたりの最大サンプル数 [default: 1000]
    -b, --batch-size SIZE       バッチサイズ [default: 16]
    -d, --download              ClinVarデータをダウンロード
    -e, --eval-only             評価のみ実行（データ準備をスキップ）
    -v, --visualize-only        可視化のみ実行
    -h, --help                  このヘルプを表示

例:
    $0 --download --model-size medium --max-samples 2000
    $0 --eval-only --model-size large
    $0 --visualize-only
EOF
}

# パラメータ解析
DOWNLOAD=false
EVAL_ONLY=false
VISUALIZE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        -s|--sequence-length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        -n|--max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -d|--download)
            DOWNLOAD=true
            shift
            ;;
        -e|--eval-only)
            EVAL_ONLY=true
            shift
            ;;
        -v|--visualize-only)
            VISUALIZE_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# ディレクトリ作成
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$PROJECT_ROOT/logs"

echo "=== ClinVar評価パイプライン開始 ==="
echo "モデルサイズ: $MODEL_SIZE"
echo "配列長: $SEQUENCE_LENGTH"
echo "最大サンプル数: $MAX_SAMPLES"
echo "バッチサイズ: $BATCH_SIZE"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo ""

# Python環境の設定
cd "$PROJECT_ROOT"

# 可視化のみの場合
if [[ "$VISUALIZE_ONLY" == true ]]; then
    echo "=== 可視化実行 ==="
    
    RESULTS_FILE="$OUTPUT_DIR/evaluation_results.json"
    if [[ ! -f "$RESULTS_FILE" ]]; then
        echo "エラー: 評価結果ファイルが見つかりません: $RESULTS_FILE"
        exit 1
    fi
    
    python scripts/clinvar_visualization.py \
        --results_file "$RESULTS_FILE" \
        --output_dir "$OUTPUT_DIR/visualizations" \
        --html_report
    
    echo "可視化完了: $OUTPUT_DIR/visualizations/"
    exit 0
fi

# 1. データ準備（評価のみでない場合）
if [[ "$EVAL_ONLY" != true ]]; then
    echo "=== データ準備フェーズ ==="
    
    if [[ "$DOWNLOAD" == true ]]; then
        echo "ClinVarデータをダウンロード中..."
        python scripts/clinvar_data_preparation.py \
            --download \
            --output_dir "$DATA_DIR" \
            --max_samples "$MAX_SAMPLES" \
            --sequence_length "$SEQUENCE_LENGTH"
    else
        echo "サンプルClinVarデータを作成中..."
        python scripts/clinvar_data_preparation.py \
            --output_dir "$DATA_DIR" \
            --max_samples "$MAX_SAMPLES" \
            --sequence_length "$SEQUENCE_LENGTH"
    fi
    
    echo "データ準備完了"
fi

# 2. モデル評価
if [[ "$VISUALIZE_ONLY" != true ]]; then
    echo "=== モデル評価フェーズ ==="
    
    # モデルパスの構築
    MODEL_PATH="$MODELS_DIR/genome_sequence-$MODEL_SIZE/ckpt.pt"
    
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "エラー: モデルファイルが見つかりません: $MODEL_PATH"
        echo "利用可能なモデル:"
        find "$MODELS_DIR" -name "ckpt.pt" 2>/dev/null || echo "  モデルが見つかりません"
        exit 1
    fi
    
    # ClinVarデータファイルの確認
    CLINVAR_DATA="$DATA_DIR/random_2000_clinvar.csv"
    if [[ ! -f "$CLINVAR_DATA" ]]; then
        echo "エラー: ClinVarデータが見つかりません: $CLINVAR_DATA"
        echo "まずデータ準備を実行してください"
        exit 1
    fi
    
    echo "モデル評価を実行中..."
    echo "モデル: $MODEL_PATH"
    echo "データ: $CLINVAR_DATA"
    
    python scripts/clinvar_evaluation.py \
        --model_path "$MODEL_PATH" \
        --clinvar_data "$CLINVAR_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE"
    
    echo "モデル評価完了"
fi

# 3. 結果可視化
echo "=== 可視化フェーズ ==="

RESULTS_FILE="$OUTPUT_DIR/evaluation_results.json"
if [[ ! -f "$RESULTS_FILE" ]]; then
    echo "エラー: 評価結果ファイルが見つかりません: $RESULTS_FILE"
    exit 1
fi

python scripts/clinvar_visualization.py \
    --results_file "$RESULTS_FILE" \
    --output_dir "$OUTPUT_DIR/visualizations" \
    --html_report

echo "可視化完了"

# 4. 結果サマリー
echo ""
echo "=== 評価結果サマリー ==="

if command -v python3 &> /dev/null; then
    python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)

print(f'Accuracy: {results[\"accuracy\"]:.3f}')
print(f'Precision: {results[\"precision\"]:.3f}')
print(f'Recall: {results[\"recall\"]:.3f}')
print(f'F1-Score: {results[\"f1_score\"]:.3f}')
print(f'ROC-AUC: {results[\"roc_auc\"]:.3f}')
print(f'PR-AUC: {results[\"pr_auc\"]:.3f}')
"
fi

echo ""
echo "=== 出力ファイル ==="
echo "評価結果: $OUTPUT_DIR/evaluation_results.json"
echo "詳細レポート: $OUTPUT_DIR/evaluation_report.txt"
echo "可視化結果: $OUTPUT_DIR/visualizations/"
echo "HTMLレポート: $OUTPUT_DIR/visualizations/evaluation_report.html"

echo ""
echo "=== ClinVar評価パイプライン完了 ==="
