#!/bin/bash

#
# OMIM評価パイプライン実行スクリプト
#
# このスクリプトは、OMIMデータの取得から評価、可視化までの
# 全プロセスを自動で実行します。
#
# 使用方法:
#   ./run_omim_evaluation.sh [options]
#
# オプション:
#   --model_size small|medium|large  モデルサイズ (デフォルト: small)
#   --sequence_length NUMBER         配列長 (デフォルト: 100)
#   --max_samples NUMBER             最大サンプル数 (デフォルト: 1000)
#   --batch_size NUMBER              バッチサイズ (デフォルト: 16)
#   --output_dir PATH                出力ディレクトリ (デフォルト: omim_evaluation_results)
#
# 例:
#   ./run_omim_evaluation.sh --model_size medium --sequence_length 200 --max_samples 2000
#

set -e  # エラー時に停止

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# デフォルト設定
MODEL_SIZE="small"
SEQUENCE_LENGTH=100
MAX_SAMPLES=1000
BATCH_SIZE=16
OUTPUT_DIR="$PROJECT_ROOT/omim_evaluation_results"

# 引数パース
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --sequence_length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "OMIM Evaluation Pipeline"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_size SIZE         Model size (small|medium|large, default: small)"
            echo "  --sequence_length LENGTH  Sequence length (default: 100)"
            echo "  --max_samples NUMBER      Maximum samples to generate (default: 1000)"
            echo "  --batch_size SIZE         Batch size for evaluation (default: 16)"
            echo "  --output_dir PATH         Output directory (default: omim_evaluation_results)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model_size medium --sequence_length 200"
            echo "  $0 --max_samples 2000 --batch_size 32"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 設定表示
echo "=== OMIM評価パイプライン開始 ==="
echo "モデルサイズ: $MODEL_SIZE"
echo "配列長: $SEQUENCE_LENGTH"
echo "最大サンプル数: $MAX_SAMPLES"
echo "バッチサイズ: $BATCH_SIZE"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo ""

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# パス設定
MODEL_PATH="$PROJECT_ROOT/gpt2-output/genome_sequence-$MODEL_SIZE/ckpt.pt"
DATA_PATH="$OUTPUT_DIR/data/omim_evaluation_dataset.csv"

# モデルファイル存在チェック
if [ ! -f "$MODEL_PATH" ]; then
    echo "エラー: モデルファイルが見つかりません: $MODEL_PATH"
    echo "利用可能なモデル:"
    ls -la "$PROJECT_ROOT/gpt2-output/" 2>/dev/null || echo "  gpt2-outputディレクトリが存在しません"
    exit 1
fi

# Python環境チェック
if ! command -v python &> /dev/null; then
    echo "エラー: Pythonが見つかりません"
    exit 1
fi

# 必要なPythonパッケージチェック
echo "Pythonパッケージをチェック中..."
python -c "
import sys
required_packages = ['pandas', 'numpy', 'torch', 'sentencepiece', 'sklearn', 'matplotlib', 'seaborn']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'エラー: 以下のパッケージが不足しています: {missing_packages}')
    print('以下のコマンドでインストールしてください:')
    print(f'pip install {\" \".join(missing_packages)}')
    sys.exit(1)
else:
    print('必要なパッケージはすべて利用可能です')
"

if [ $? -ne 0 ]; then
    echo "パッケージチェックに失敗しました"
    exit 1
fi

# データ準備フェーズ
echo "=== データ準備フェーズ ==="
echo "OMIMサンプルデータを作成中..."

cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/omim_data_preparation.py" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$MAX_SAMPLES" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --seed 42

if [ $? -ne 0 ]; then
    echo "エラー: データ準備に失敗しました"
    exit 1
fi

echo "データ準備完了"

# データファイル存在チェック
if [ ! -f "$DATA_PATH" ]; then
    echo "エラー: データファイルが生成されませんでした: $DATA_PATH"
    exit 1
fi

# モデル評価フェーズ
echo "=== モデル評価フェーズ ==="
echo "モデル評価を実行中..."
echo "モデル: $MODEL_PATH"
echo "データ: $DATA_PATH"

python "$SCRIPT_DIR/omim_evaluation.py" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE"

if [ $? -ne 0 ]; then
    echo "エラー: モデル評価に失敗しました"
    exit 1
fi

echo "モデル評価完了"

# 可視化フェーズ
echo "=== 可視化フェーズ ==="
echo "評価結果の可視化を実行中..."

python "$SCRIPT_DIR/omim_visualization.py" \
    --results_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "エラー: 可視化に失敗しました"
    exit 1
fi

echo "可視化完了"

# 結果表示
echo ""
echo "=== 評価結果サマリー ==="

# JSONファイルから結果を抽出
RESULTS_FILE="$OUTPUT_DIR/omim_evaluation_results.json"
if [ -f "$RESULTS_FILE" ]; then
    python -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)

print(f'Accuracy: {results.get(\"accuracy\", 0):.3f}')
print(f'Precision: {results.get(\"precision\", 0):.3f}')
print(f'Recall: {results.get(\"recall\", 0):.3f}')
print(f'F1-Score: {results.get(\"f1_score\", 0):.3f}')
print(f'ROC-AUC: {results.get(\"roc_auc\", 0):.3f}')
print(f'PR-AUC: {results.get(\"pr_auc\", 0):.3f}')
print(f'Sensitivity: {results.get(\"sensitivity\", 0):.3f}')
print(f'Specificity: {results.get(\"specificity\", 0):.3f}')
"
else
    echo "結果ファイルが見つかりません: $RESULTS_FILE"
fi

echo ""
echo "=== 出力ファイル ==="
echo "評価結果: $OUTPUT_DIR/omim_evaluation_results.json"
echo "詳細レポート: $OUTPUT_DIR/omim_evaluation_report.txt"
echo "可視化結果: $OUTPUT_DIR/visualizations/"
echo "HTMLレポート: $OUTPUT_DIR/visualizations/omim_evaluation_report.html"

echo ""
echo "=== OMIM評価パイプライン完了 ==="

# OMIMについての情報表示
echo ""
echo "=== OMIMデータベースについて ==="
cat << 'EOF'
OMIM (Online Mendelian Inheritance in Man) は遺伝性疾患の包括的データベースです。
- 25,000以上の遺伝性疾患・遺伝子情報を収録
- 単一遺伝子疾患、複合遺伝、染色体異常の詳細な分類
- 遺伝形式(常染色体優性/劣性、X連鎖、ミトコンドリア)の明確な分類
- 実際のOMIMデータ使用には登録が必要: https://omim.org/
- 現在はサンプルデータによる概念実証を実装
EOF

echo ""
echo "パイプライン実行が完了しました！"
