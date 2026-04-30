#!/bin/bash

#
# RNA Benchmark 評価パイプライン実行スクリプト
#
# ProteinGymの構成を踏襲し、データ準備 → 評価を一括実行します。
#

set -e
trap 'echo "エラー: $BASH_SOURCE:$LINENO でコマンドが失敗しました" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -z "$LEARNING_SOURCE_DIR" ]; then
    echo "エラー: LEARNING_SOURCE_DIR環境変数が設定されていません"
    exit 1
fi

if [ -z "$EVALUATION_OUTPUT_DIR" ]; then
    EVALUATION_OUTPUT_DIR="$LEARNING_SOURCE_DIR"
    echo "EVALUATION_OUTPUT_DIRが未設定のため、LEARNING_SOURCE_DIRを使用します"
fi

# デフォルト設定
MODEL_TYPE="bert"
MODEL_PATH=""
BENCHMARK_DIR="$EVALUATION_OUTPUT_DIR/rna/data/benchmark"
DATASETS=""
MAX_CELLS_PER_DATASET=10000
BATCH_SIZE=16
DEVICE="cuda"
SKIP_DATA_PREP=false
SKIP_EVALUATION=false
GENE_SYMBOL_MAP=""
SYMBOL_COLUMN="symbol"
ENSEMBL_COLUMN="ensembl_id"

OUTPUT_DIR="$EVALUATION_OUTPUT_DIR/rna/report/rna_benchmark"
DATA_DIR="$OUTPUT_DIR/data"
DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            DATA_DIR="$OUTPUT_DIR/data"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --benchmark_dir)
            BENCHMARK_DIR="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --max_cells_per_dataset)
            MAX_CELLS_PER_DATASET="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gene_symbol_map)
            GENE_SYMBOL_MAP="$2"
            shift 2
            ;;
        --symbol_column)
            SYMBOL_COLUMN="$2"
            shift 2
            ;;
        --ensembl_column)
            ENSEMBL_COLUMN="$2"
            shift 2
            ;;
        --skip_data_prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --skip_evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        -h|--help)
            echo "RNA Benchmark Evaluation Pipeline"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -o, --output-dir PATH     Output directory (default: \$LEARNING_SOURCE_DIR/rna/report/rna_benchmark)"
            echo "  --model_type TYPE         Model type (bert|gpt2)"
            echo "  --model_path PATH         Model checkpoint path"
            echo "  --benchmark_dir PATH      Benchmark .h5ad directory"
            echo "  --data_path PATH          既存のRNAベンチマークJSONL"
            echo "  --datasets LIST           Dataset names (comma separated)"
            echo "  --max_cells_per_dataset N Max cells per dataset"
            echo "  --batch_size N            Batch size"
            echo "  --device DEVICE           cuda or cpu"
            echo "  --gene_symbol_map PATH    Gene symbol -> Ensembl mapping TSV/CSV"
            echo "  --symbol_column NAME      Symbol column name in mapping file"
            echo "  --ensembl_column NAME     Ensembl column name in mapping file"
            echo "  --skip_data_prep          Skip data preparation"
            echo "  --skip_evaluation         Skip evaluation"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== RNA Benchmark評価パイプライン開始 ==="
echo "モデル種別: $MODEL_TYPE"
echo "モデルパス: $MODEL_PATH"
echo "ベンチマークデータ: $BENCHMARK_DIR"
echo "データセット: ${DATASETS:-ALL}"
echo "最大細胞数/データセット: $MAX_CELLS_PER_DATASET"
echo "バッチサイズ: $BATCH_SIZE"
echo "デバイス: $DEVICE"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "データディレクトリ: $DATA_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

DATASET_PATH="$DATA_DIR/rna_benchmark_dataset.jsonl"
if [ -n "$DATA_PATH" ]; then
    DATASET_PATH="$DATA_PATH"
fi

if [ "$SKIP_DATA_PREP" = false ]; then
    echo "=== データ準備フェーズ ==="
    cd "$PROJECT_ROOT"

    python molcrawl/tasks/evaluation/rna_benchmark/data_preparation.py \
        --benchmark_dir "$BENCHMARK_DIR" \
        --output_dir "$DATA_DIR" \
        --datasets "$DATASETS" \
        --max_cells_per_dataset "$MAX_CELLS_PER_DATASET" \
        ${GENE_SYMBOL_MAP:+--gene_symbol_map "$GENE_SYMBOL_MAP"} \
        --symbol_column "$SYMBOL_COLUMN" \
        --ensembl_column "$ENSEMBL_COLUMN"
fi

if [ "$SKIP_EVALUATION" = false ]; then
    if [ ! -f "$DATASET_PATH" ]; then
        # skip_data_prep時は共通データ出力を参照できるようにする
        FALLBACK_DATA_PATH="$EVALUATION_OUTPUT_DIR/rna/report/rna_benchmark/data/rna_benchmark_dataset.jsonl"
        if [ -f "$FALLBACK_DATA_PATH" ]; then
            DATASET_PATH="$FALLBACK_DATA_PATH"
        else
            echo "エラー: データセットが見つかりません: $DATASET_PATH"
            echo "  代替候補も見つかりません: $FALLBACK_DATA_PATH"
            echo "  先にデータ準備を実行するか --data_path を指定してください"
            exit 1
        fi
    fi

    if [ -z "$MODEL_PATH" ]; then
        echo "エラー: --model_path が指定されていません"
        exit 1
    fi

    echo "=== 評価フェーズ ==="
    cd "$PROJECT_ROOT"

    python molcrawl/tasks/evaluation/rna_benchmark/evaluation.py \
        --model_type "$MODEL_TYPE" \
        --model_path "$MODEL_PATH" \
        --data_path "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --datasets "$DATASETS" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE"
fi

echo "=== RNA Benchmark評価パイプライン完了 ==="
