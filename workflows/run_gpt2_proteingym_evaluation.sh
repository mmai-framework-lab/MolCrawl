#!/bin/bash

#
# GPT-2 ProteinGym評価パイプライン実行スクリプト
#
# このスクリプトは、GPT-2モデルを使用したProteinGymデータの評価パイプラインを実行します。
# データ準備から評価、可視化までの全プロセスを自動で実行します。
#
# 注意: このスクリプトはworkflows/ディレクトリから実行されることを想定しています
#
# Usage: ./run_gpt2_proteingym_evaluation.sh [OPTIONS]

set -e  # エラー時に停止

# エラー時に行番号を表示
trap 'echo "エラー: $BASH_SOURCE:$LINENO でコマンドが失敗しました" >&2' ERR

# スクリプトのディレクトリを取得
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

# デフォルト設定
MODEL_SIZE="small"  # デフォルトのモデルサイズ
MODELS_DIR="$PROJECT_ROOT/gpt2-output"
MODEL_PATH=""  # 空の場合はMODEL_SIZEから自動構築
DATA_PATH=""
# デフォルト出力先（-o/--output-dirで上書き可能）
OUTPUT_DIR="$EVALUATION_OUTPUT_DIR/protein_sequence/report/gpt2_proteingym"
DATA_DIR="$EVALUATION_OUTPUT_DIR/protein_sequence/data/gpt2_proteingym"  # データ準備時の出力先
BATCH_SIZE=32
DEVICE="cuda"
TOKENIZER_PATH=""
CREATE_SAMPLE=false
DOWNLOAD_DATA=false
VISUALIZE=false

# ヘルプメッセージ
show_help() {
    cat << EOF
GPT-2 ProteinGym Evaluation Pipeline

Usage: $0 [OPTIONS]

Options:
    -m, --model_path PATH       Path to trained GPT-2 protein sequence model (default: gpt2-output/protein_sequence-small/ckpt.pt)
    --model_size SIZE           Model size (small/medium/large/xl) for auto path (default: small)
    -d, --data_path PATH        Path to ProteinGym data file (required unless --download-data)
    -o, --output_dir PATH       Output directory (default: \$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_proteingym)
    -b, --batch_size SIZE       Batch size for evaluation (default: $BATCH_SIZE)
    --device DEVICE             Device to use: cuda or cpu (default: $DEVICE)
    --tokenizer_path PATH       Path to SentencePiece tokenizer (auto-detect if not provided)
    --create-sample             Create sample data for testing
    --download-data             Download ProteinGym data automatically
    --visualize                 Generate visualization plots
    -h, --help                  Show this help message

Examples:
    # Basic GPT-2 evaluation with default model and existing data
    $0 -d proteingym_data/sample.csv

    # Specify model size
    $0 --model_size medium -d proteingym_data/sample.csv

    # Specify custom model path
    $0 -m gpt2-output/protein_sequence-large/ckpt.pt -d proteingym_data/sample.csv

    # Download data, evaluate, and visualize
    $0 --download-data --visualize

    # Evaluation with custom batch size
    $0 -d data.csv -b 64
EOF
}

# 引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model_size)
            MODEL_SIZE="$2"
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

# モデルパスの自動構築（指定されていない場合）
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$MODELS_DIR/protein_sequence-$MODEL_SIZE/ckpt.pt"
    echo "モデルパスが指定されていません。デフォルトを使用: $MODEL_PATH"
fi

# モデルファイルの存在チェック
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "エラー: モデルファイルが見つかりません: $MODEL_PATH"
    echo "利用可能なモデル:"
    find "$MODELS_DIR" -name "ckpt.pt" 2>/dev/null || echo "  モデルが見つかりません"
    exit 1
fi

if [[ "$CREATE_SAMPLE" == false && "$DOWNLOAD_DATA" == false && -z "$DATA_PATH" ]]; then
    echo "エラー: データパスが必要です (-d/--data_path)"
    echo "または --create-sample または --download-data を指定してください"
    show_help
    exit 1
fi

# ログディレクトリの作成
mkdir -p logs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

echo "=== GPT-2 ProteinGym評価パイプライン開始 ==="
echo "モデル: $MODEL_PATH"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "データディレクトリ: $DATA_DIR"
echo "デバイス: $DEVICE"
echo "バッチサイズ: $BATCH_SIZE"
echo ""

# Python環境の確認
if ! command -v python &> /dev/null; then
    echo "エラー: Pythonが見つかりません"
    exit 1
fi

# 必要なPythonパッケージの確認（GPT-2評価用）
echo "Pythonパッケージを確認中..."
 -c "
import sys
required_packages = ['torch', 'numpy', 'pandas', 'sklearn', 'sentencepiece', 'scipy']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'エラー: 必要なパッケージが見つかりません: {missing}')
    print('以下のコマンドでインストールしてください: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('全ての必要なパッケージが見つかりました。')
"

if [[ $? -ne 0 ]]; then
    exit 1
fi

echo "GPT-2環境変数設定: LEARNING_SOURCE_DIR=$LEARNING_SOURCE_DIR"
echo ""

# データのダウンロード（必要な場合）
if [[ "$DOWNLOAD_DATA" == true ]]; then
    echo "=== データ準備フェーズ ==="
    echo "推奨されるProteinGymデータセットをダウンロード中..."

    cd "$PROJECT_ROOT"

    python molcrawl/tasks/evaluation/proteingym/gpt2_data_preparation.py \
        --download recommended \
        --data_dir "$DATA_DIR"

    # ダウンロードされたアッセイファイルを探す
    FIRST_ASSAY=$(find "$DATA_DIR" -name "*.csv" -path "*/DMS_ProteinGym_substitutions/*" | head -1)
    if [[ -n "$FIRST_ASSAY" ]]; then
        DATA_PATH="$FIRST_ASSAY"
        echo "ダウンロードされたデータを使用: $DATA_PATH"
    else
        # フォールバック：任意のアッセイファイル
        FIRST_ASSAY=$(find "$DATA_DIR" -name "*.csv" | head -1)
        if [[ -n "$FIRST_ASSAY" ]]; then
            DATA_PATH="$FIRST_ASSAY"
            echo "ダウンロードされたデータを使用: $DATA_PATH"
        else
            echo "エラー: ダウンロード後にデータファイルが見つかりません"
            exit 1
        fi
    fi
    echo ""
fi

# サンプルデータの作成（必要な場合）
if [[ "$CREATE_SAMPLE" == true ]]; then
    echo "=== データ準備フェーズ ==="
    echo "テスト用サンプルデータを準備中..."

    cd "$PROJECT_ROOT"

    # まず、推奨データセットをダウンロード
    python molcrawl/tasks/evaluation/proteingym/gpt2_data_preparation.py \
        --data_dir "$DATA_DIR" \
        --download recommended

    if [[ $? -ne 0 ]]; then
        echo "警告: データダウンロードに失敗しました。ローカルデータを探します..."
    fi

    # ダウンロードされたアッセイファイルを探す
    FIRST_ASSAY=$(find "$DATA_DIR" -name "*.csv" -path "*/DMS_ProteinGym_substitutions/*" | head -1)

    if [[ -n "$FIRST_ASSAY" ]]; then
        DATA_PATH="$FIRST_ASSAY"
        echo "✓ ダウンロードされたデータを使用: $DATA_PATH"
    else
        # フォールバック：任意のアッセイファイル
        FIRST_ASSAY=$(find "$DATA_DIR" -name "*.csv" | head -1)
        if [[ -n "$FIRST_ASSAY" ]]; then
            DATA_PATH="$FIRST_ASSAY"
            echo "✓ 既存データを使用: $DATA_PATH"
        else
            echo "エラー: サンプルデータが見つかりません"
            echo "手動でデータをダウンロードするか、--data_pathでデータファイルを指定してください"
            exit 1
        fi
    fi
    echo ""
fi

# モデルファイルの存在確認
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "エラー: GPT-2モデルファイルが見つかりません: $MODEL_PATH"
    exit 1
fi

# データファイルの存在確認
if [[ ! -f "$DATA_PATH" ]]; then
    echo "エラー: データファイルが見つかりません: $DATA_PATH"
    exit 1
fi

# =============================================================================
# モデル評価フェーズ
# =============================================================================

echo "=== モデル評価フェーズ ==="
echo "GPT-2 ProteinGym評価を実行中..."
echo "データファイル: $DATA_PATH"

cd "$PROJECT_ROOT"

EVAL_ARGS=(
    "molcrawl/tasks/evaluation/proteingym/gpt2_evaluation.py"
    "--model_path" "$MODEL_PATH"
    "--proteingym_data" "$DATA_PATH"
    "--output_dir" "$OUTPUT_DIR"
    "--batch_size" "$BATCH_SIZE"
    "--device" "$DEVICE"
)

if [[ -n "$TOKENIZER_PATH" ]]; then
    EVAL_ARGS+=("--tokenizer_path" "$TOKENIZER_PATH")
fi

 "${EVAL_ARGS[@]}"

if [[ $? -ne 0 ]]; then
    echo "エラー: 評価に失敗しました"
    exit 1
fi

echo "評価完了"
echo ""

# =============================================================================
# 可視化フェーズ
# =============================================================================

if [[ "$VISUALIZE" == true ]]; then
    echo "=== 可視化フェーズ ==="
    echo "GPT-2評価結果の可視化を実行中..."

    # 最新の評価結果ディレクトリを探す
    LATEST_RESULT_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*gpt2_proteingym*" | sort | tail -1)

    if [ -z "$LATEST_RESULT_DIR" ]; then
        echo "警告: 評価結果ディレクトリが見つかりません"
        RESULTS_FILE="$OUTPUT_DIR/evaluation_results.json"
        VIS_DIR="$OUTPUT_DIR/visualizations"
    else
        RESULTS_FILE="$LATEST_RESULT_DIR/evaluation_results.json"
        VIS_DIR="$LATEST_RESULT_DIR/visualizations"
    fi

    if [[ -f "$RESULTS_FILE" ]]; then
        python "$PROJECT_ROOT/molcrawl/tasks/evaluation/proteingym/gpt2_visualization.py" \
            --results_file "$RESULTS_FILE" \
            --output_dir "$VIS_DIR"

        if [[ $? -eq 0 ]]; then
            echo "可視化完了: $VIS_DIR"
        else
            echo "警告: 可視化に失敗しました"
        fi
    else
        echo "警告: 可視化用の結果ファイルが見つかりません: $RESULTS_FILE"
    fi
    echo ""
fi

# =============================================================================
# 完了メッセージ
# =============================================================================

echo ""
echo "=== GPT-2 ProteinGym評価パイプライン完了 ==="
if [[ -f "$OUTPUT_DIR/evaluation_report.txt" ]]; then
    echo "評価レポート:"
    cat "$OUTPUT_DIR/evaluation_report.txt"
else
    echo "レポートファイルが見つかりません。"
fi

echo ""
echo "=== 出力ファイル ==="
ls -la "$OUTPUT_DIR" 2>/dev/null || echo "出力ディレクトリ: $OUTPUT_DIR"

echo ""
echo "✅ 全ての処理が正常に完了しました"
echo "📁 結果保存先: $OUTPUT_DIR"
