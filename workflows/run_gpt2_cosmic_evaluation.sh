#!/bin/bash
"""
GPT-2 COSMIC評価パイプライン実行スクリプト

このスクリプトは、COSMICデータの取得から評価、可視化までの
全プロセスを自動で実行します。

注意: このスクリプトはworkflows/ディレクトリから実行されることを想定しています
"""

set -e  # エラー時に停止

# エラー時に行番号を表示
trap 'echo "エラー: $BASH_SOURCE:$LINENO でコマンドが失敗しました" >&2' ERR

# 設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"  # プロジェクトルートディレクトリ

# LEARNING_SOURCE_DIRの確認
if [ -z "$LEARNING_SOURCE_DIR" ]; then
    echo "エラー: LEARNING_SOURCE_DIR環境変数が設定されていません"
    echo "実行前に以下を設定してください:"
    echo "  export LEARNING_SOURCE_DIR=/path/to/learning_source"
    exit 1
fi

# EVALUATION_OUTPUT_DIRの確認（デフォルトは$LEARNING_SOURCE_DIRと同じ）
if [ -z "$EVALUATION_OUTPUT_DIR" ]; then
    EVALUATION_OUTPUT_DIR="$LEARNING_SOURCE_DIR"
    echo "EVALUATION_OUTPUT_DIRが未設定のため、LEARNING_SOURCE_DIRを使用します"
fi

# デフォルト出力先（-o/--output-dirで上書き可能）
OUTPUT_DIR="$EVALUATION_OUTPUT_DIR/genome_sequence/report/cosmic_evaluation"
DATA_DIR="$EVALUATION_OUTPUT_DIR/genome_sequence/data/cosmic"  # データ準備時の出力先
MODELS_DIR="$PROJECT_ROOT/gpt2-output"

# デフォルト設定
MODEL_SIZE="small"
SEQUENCE_LENGTH=100
MAX_SAMPLES=1000
BATCH_SIZE=16
TOKENIZER_PATH=""  # 空の場合は自動検出

# ヘルプ表示
show_help() {
    cat << EOF
GPT-2 COSMIC評価パイプライン

使用法: $0 [オプション]

オプション:
    -o, --output-dir PATH       出力ディレクトリ [default: \$LEARNING_SOURCE_DIR/genome_sequence/report/cosmic_evaluation]
    -m, --model-size SIZE       モデルサイズ (small/medium/large/xl) [default: small]
    -t, --tokenizer PATH        トークナイザーパス（指定しない場合は自動検出）
    -s, --sequence-length LEN   配列長 [default: 100]
    -n, --max-samples NUM       クラスあたりの最大サンプル数 [default: 1000]
    -b, --batch-size SIZE       バッチサイズ [default: 16]
    -d, --download              COSMICデータをダウンロード（現在はサンプルデータのみ）
    -e, --eval-only             評価のみ実行（データ準備をスキップ）
    -v, --visualize-only        可視化のみ実行
    -h, --help                  このヘルプを表示

例:
    # デフォルト出力先での実行
    $0 --model-size medium --max-samples 2000

    # カスタム出力ディレクトリを指定
    $0 --eval-only -o /custom/output/cosmic_results

    # 可視化のみ実行
    $0 --visualize-only --output-dir ./my_results

注意:
    実際のCOSMICデータのダウンロードには登録が必要です。
    現在はサンプルデータを使用した評価のみ対応しています。
EOF
}

# パラメータ解析
DOWNLOAD=false
EVAL_ONLY=false
VISUALIZE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        -t|--tokenizer)
            TOKENIZER_PATH="$2"
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

echo "=== COSMIC評価パイプライン開始 ==="
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

    RESULTS_FILE="$OUTPUT_DIR/cosmic_evaluation_results.json"
    if [[ ! -f "$RESULTS_FILE" ]]; then
        echo "エラー: 評価結果ファイルが見つかりません: $RESULTS_FILE"
        exit 1
    fi

    python "$PROJECT_ROOT/scripts/evaluation/gpt2/cosmic_visualization.py" \
        --result-dir "$OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR/visualizations"

    echo "可視化完了: $OUTPUT_DIR/visualizations/"
    exit 0
fi

# 1. データ準備（評価のみでない場合）
if [[ "$EVAL_ONLY" != true ]]; then
    echo "=== データ準備フェーズ ==="

    echo "COSMICサンプルデータを作成中..."
    python "$PROJECT_ROOT/scripts/evaluation/gpt2/cosmic_data_preparation.py" \
        --output_dir "$DATA_DIR" \
        --max_samples "$MAX_SAMPLES" \
        --create_sample_data

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

    # COSMICデータファイルの確認
    COSMIC_DATA="$DATA_DIR/cosmic_evaluation_dataset.csv"
    if [[ ! -f "$COSMIC_DATA" ]]; then
        echo "エラー: COSMICデータが見つかりません: $COSMIC_DATA"
        echo "まずデータ準備を実行してください"
        exit 1
    fi

    echo "モデル評価を実行中..."
    echo "モデル: $MODEL_PATH"
    echo "データ: $COSMIC_DATA"

    # Pythonコマンド引数を準備
    EVAL_ARGS=(
        "$PROJECT_ROOT/scripts/evaluation/gpt2/cosmic_evaluation.py"
        --model_path "$MODEL_PATH"
        --cosmic_data "$COSMIC_DATA"
        --output_dir "$OUTPUT_DIR"
        --batch_size "$BATCH_SIZE"
    )

    # トークナイザーパスが指定されている場合は追加
    if [[ -n "$TOKENIZER_PATH" ]]; then
        EVAL_ARGS+=(--tokenizer_path "$TOKENIZER_PATH")
        echo "トークナイザー: $TOKENIZER_PATH"
    else
        echo "トークナイザー: 自動検出"
    fi

    python "${EVAL_ARGS[@]}"

    echo "モデル評価完了"
fi

# 3. 結果可視化
echo "=== 可視化フェーズ ==="

RESULTS_FILE="$OUTPUT_DIR/cosmic_evaluation_results.json"
if [[ ! -f "$RESULTS_FILE" ]]; then
    echo "エラー: 評価結果ファイルが見つかりません: $RESULTS_FILE"
    exit 1
fi

 "$PROJECT_ROOT/scripts/evaluation/gpt2/cosmic_visualization.py" \
    --results_file "$RESULTS_FILE" \
    --output_dir "$OUTPUT_DIR/visualizations"

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
print(f'Sensitivity: {results[\"sensitivity\"]:.3f}')
print(f'Specificity: {results[\"specificity\"]:.3f}')
"
fi

echo ""
echo "=== 出力ファイル ==="
echo "評価結果: $OUTPUT_DIR/cosmic_evaluation_results.json"
echo "詳細レポート: $OUTPUT_DIR/cosmic_evaluation_report.txt"
echo "可視化結果: $OUTPUT_DIR/visualizations/"
echo "HTMLレポート: $OUTPUT_DIR/visualizations/cosmic_evaluation_report.html"

echo ""
echo "=== COSMIC評価パイプライン完了 ==="

# COSMICデータベースについての補足情報
echo ""
echo "=== COSMICデータベースについて ==="
echo "COSMIC (Catalogue of Somatic Mutations in Cancer) は世界最大の癌体細胞変異データベースです。"
echo "- 600万以上の癌関連変異を収録"
echo "- 癌の種類、遺伝子、変異タイプ別の詳細な分類"
echo "- 実際のCOSMICデータ使用には登録が必要: https://cancer.sanger.ac.uk/cosmic"
echo "- 現在はサンプルデータによる概念実証を実装"
