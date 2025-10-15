#!/bin/bash

# ProteinGym評価スクリプト実行用シェルスクリプト
# Usage: ./run_proteingym_evaluation.sh [OPTIONS]

set -e  # エラー時に停止

# デフォルト設定
MODEL_PATH=""
DATA_PATH=""
OUTPUT_DIR="./proteingym_evaluation_results"
BATCH_SIZE=32
DEVICE="cuda"
TOKENIZER_PATH=""
CREATE_SAMPLE=false
DOWNLOAD_DATA=false
VISUALIZE=false

# ヘルプメッセージ
show_help() {
    cat << EOF
ProteinGym Evaluation Script

Usage: $0 [OPTIONS]

Options:
    -m, --model_path PATH       Path to trained protein sequence model (required)
    -d, --data_path PATH        Path to ProteinGym data file (required unless --download-data)
    -o, --output_dir PATH       Output directory (default: $OUTPUT_DIR)
    -b, --batch_size SIZE       Batch size for evaluation (default: $BATCH_SIZE)
    --device DEVICE             Device to use: cuda or cpu (default: $DEVICE)
    --tokenizer_path PATH       Path to tokenizer (auto-detect if not provided)
    --create-sample             Create sample data for testing
    --download-data             Download ProteinGym data automatically
    --visualize                 Generate visualization plots
    -h, --help                  Show this help message

Examples:
    # Basic evaluation with existing data
    $0 -m gpt2-output/protein_sequence-small/ckpt.pt -d proteingym_data/sample.csv

    # Create sample data and evaluate
    $0 -m gpt2-output/protein_sequence-small/ckpt.pt --create-sample -d sample_data.csv

    # Download data, evaluate, and visualize
    $0 -m gpt2-output/protein_sequence-small/ckpt.pt --download-data --visualize

    # CPU evaluation with custom settings
    $0 -m model.pt -d data.csv --device cpu -b 16 -o results/
EOF
}

# 引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --tokenizer_path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --create-sample)
            CREATE_SAMPLE=true
            shift
            ;;
        --download-data)
            DOWNLOAD_DATA=true
            shift
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# 必要な引数のチェック
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: Model path is required (-m/--model_path)"
    show_help
    exit 1
fi

if [[ "$CREATE_SAMPLE" == false && "$DOWNLOAD_DATA" == false && -z "$DATA_PATH" ]]; then
    echo "Error: Data path is required (-d/--data_path) unless --create-sample or --download-data is specified"
    show_help
    exit 1
fi

# ログディレクトリの作成
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "=== ProteinGym Evaluation Pipeline ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo ""

# Python環境の確認
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# 必要なPythonパッケージの確認
echo "Checking Python dependencies..."
python -c "
import sys
required_packages = ['torch', 'numpy', 'pandas', 'sklearn', 'sentencepiece', 'scipy']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'Error: Missing required packages: {missing}')
    print('Please install them using: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('All required packages found.')
"

if [[ $? -ne 0 ]]; then
    exit 1
fi

# データのダウンロード（必要な場合）
if [[ "$DOWNLOAD_DATA" == true ]]; then
    echo "Downloading recommended ProteinGym datasets..."
    python scripts/proteingym_data_preparation.py --download recommended --data_dir proteingym_data/
    
    # ダウンロードされたアッセイファイルを探す
    FIRST_ASSAY=$(find proteingym_data/ -name "*.csv" -path "*/DMS_ProteinGym_substitutions/*" | head -1)
    if [[ -n "$FIRST_ASSAY" ]]; then
        DATA_PATH="$FIRST_ASSAY"
        echo "Using downloaded data: $DATA_PATH"
    else
        # フォールバック：任意のアッセイファイル
        FIRST_ASSAY=$(find proteingym_data/ -name "*.csv" | head -1)
        if [[ -n "$FIRST_ASSAY" ]]; then
            DATA_PATH="$FIRST_ASSAY"
            echo "Using downloaded data: $DATA_PATH"
        else
            echo "Error: No data files found after download"
            exit 1
        fi
    fi
fi

# サンプルデータの作成（必要な場合）
if [[ "$CREATE_SAMPLE" == true ]]; then
    if [[ -z "$DATA_PATH" ]]; then
        DATA_PATH="sample_proteingym_data.csv"
    fi
    
    echo "Creating sample data: $DATA_PATH"
    python scripts/proteingym_evaluation.py \
        --model_path "$MODEL_PATH" \
        --proteingym_data "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --create_sample_data \
        --device "$DEVICE"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to create sample data"
        exit 1
    fi
fi

# モデルファイルの存在確認
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# データファイルの存在確認
if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# 評価の実行
echo "Starting ProteinGym evaluation..."
echo "Data file: $DATA_PATH"

EVAL_CMD="python scripts/proteingym_evaluation.py \
    --model_path \"$MODEL_PATH\" \
    --proteingym_data \"$DATA_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --batch_size $BATCH_SIZE \
    --device \"$DEVICE\""

if [[ -n "$TOKENIZER_PATH" ]]; then
    EVAL_CMD="$EVAL_CMD --tokenizer_path \"$TOKENIZER_PATH\""
fi

echo "Running: $EVAL_CMD"
eval $EVAL_CMD

if [[ $? -ne 0 ]]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo "Evaluation completed successfully!"

# 可視化の実行（必要な場合）
if [[ "$VISUALIZE" == true ]]; then
    echo "Generating visualization plots..."
    
    RESULTS_FILE="$OUTPUT_DIR/evaluation_results.json"
    VIS_DIR="$OUTPUT_DIR/visualizations"
    
    if [[ -f "$RESULTS_FILE" ]]; then
        python scripts/proteingym_visualization.py \
            --results_file "$RESULTS_FILE" \
            --output_dir "$VIS_DIR"
        
        if [[ $? -eq 0 ]]; then
            echo "Visualization completed: $VIS_DIR"
        else
            echo "Warning: Visualization failed"
        fi
    else
        echo "Warning: Results file not found for visualization: $RESULTS_FILE"
    fi
fi

# 結果の表示
echo ""
echo "=== Results Summary ==="
if [[ -f "$OUTPUT_DIR/evaluation_report.txt" ]]; then
    echo "Evaluation report:"
    cat "$OUTPUT_DIR/evaluation_report.txt"
else
    echo "Report file not found."
fi

echo ""
echo "=== Output Files ==="
ls -la "$OUTPUT_DIR"

echo ""
echo "All tasks completed successfully!"
echo "Results saved in: $OUTPUT_DIR"