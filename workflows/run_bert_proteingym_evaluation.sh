#!/bin/bash

#
# BERT ProteinGym評価パイプライン実行スクリプト
#
# このスクリプトは、ProteinGymデータの準備から評価、可視化までの
# 全プロセスを自動で実行します。
#
# 注意: このスクリプトはworkflows/ディレクトリから実行されることを想定しています
#
# 使用方法:
#   ./run_bert_proteingym_evaluation.sh [options]
#
# オプション:
#   --model_path PATH                モデルパス (デフォルト: runs_train_bert_protein_sequence/checkpoint-2000)
#   --tokenizer_path PATH            トークナイザーパス (デフォルト: EsmSequenceTokenizer)
#   --max_variants NUMBER            最大バリアント数 (デフォルト: 1000)
#   --batch_size NUMBER              バッチサイズ (デフォルト: 16)
#   --device cuda|cpu                デバイス (デフォルト: cuda)
#   --download                       ProteinGymデータをダウンロード
#   --sample_only                    サンプルデータのみ作成
#   --skip_data_prep                 データ準備をスキップ
#   --skip_evaluation                評価をスキップ
#   --skip_visualization             可視化をスキップ
#
# 例:
#   ./run_bert_proteingym_evaluation.sh --max_variants 2000 --batch_size 32
#   ./run_bert_proteingym_evaluation.sh --download --sample_only
#

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
MODEL_PATH="runs_train_bert_protein_sequence/checkpoint-2000"
TOKENIZER_PATH="None"  # protein_sequenceはEsmSequenceTokenizerを使用
MAX_VARIANTS=1000
BATCH_SIZE=16
DEVICE="cuda"
DOWNLOAD=false
SAMPLE_ONLY=false
SKIP_DATA_PREP=false
SKIP_EVALUATION=false
SKIP_VISUALIZATION=false

DATA_DIR="$EVALUATION_OUTPUT_DIR/protein_sequence/data/proteingym"  # データ準備時の出力先
OUTPUT_DIR="$EVALUATION_OUTPUT_DIR/protein_sequence/report/bert_proteingym"  # デフォルト出力先（-o/--output-dirで上書き可能）

# 引数パース
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tokenizer_path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --max_variants)
            MAX_VARIANTS="$2"
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
        --download)
            DOWNLOAD=true
            shift
            ;;
        --sample_only)
            SAMPLE_ONLY=true
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
            echo "BERT ProteinGym Evaluation Pipeline"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -o, --output-dir PATH     Output directory (default: \$LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym)"
            echo "  --model_path PATH         Model path (default: runs_train_bert_protein_sequence/checkpoint-2000)"
            echo "  --tokenizer_path PATH     Tokenizer path (default: EsmSequenceTokenizer)"
            echo "  --max_variants NUMBER     Maximum variants per assay (default: 1000)"
            echo "  --batch_size SIZE         Batch size for evaluation (default: 16)"
            echo "  --device DEVICE           Device to use (cuda|cpu, default: cuda)"
            echo "  --download                Download ProteinGym data from official source"
            echo "  --sample_only             Create sample dataset only"
            echo "  --skip_data_prep          Skip data preparation phase"
            echo "  --skip_evaluation         Skip evaluation phase"
            echo "  --skip_visualization      Skip visualization phase"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic evaluation with default output"
            echo "  $0 --max_variants 2000 --batch_size 32"
            echo ""
            echo "  # Specify custom output directory"
            echo "  $0 --download --sample_only -o /custom/output/path"
            echo ""
            echo "  # Skip data preparation (use existing data)"
            echo "  $0 --skip_data_prep --output-dir ./my_results"
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
echo "=== BERT ProteinGym評価パイプライン開始 ==="
echo "モデルパス: $MODEL_PATH"
if [ "$TOKENIZER_PATH" = "None" ] || [ -z "$TOKENIZER_PATH" ]; then
    echo "トークナイザー: EsmSequenceTokenizer (built-in)"
else
    echo "トークナイザーパス: $TOKENIZER_PATH"
fi
echo "最大バリアント数: $MAX_VARIANTS"
echo "バッチサイズ: $BATCH_SIZE"
echo "デバイス: $DEVICE"
echo "データディレクトリ: $DATA_DIR"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo ""

# 出力ディレクトリ作成（既存の場合はスキップ）
if [ -e "$DATA_DIR" ] && [ ! -d "$DATA_DIR" ]; then
    echo "エラー: データディレクトリのパスが既存ファイルと衝突しています: $DATA_DIR"
    exit 1
fi
mkdir -p "$DATA_DIR"

if [ -e "$OUTPUT_DIR" ] && [ ! -d "$OUTPUT_DIR" ]; then
    echo "エラー: 出力ディレクトリのパスが既存ファイルと衝突しています: $OUTPUT_DIR"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# パス設定
DATASET_PATH="$DATA_DIR/bert_proteingym_dataset.csv"

# モデルファイル存在チェック（評価を実行する場合のみ）
if [ "$SKIP_EVALUATION" = false ]; then
    if [ ! -d "$MODEL_PATH" ]; then
        echo "エラー: モデルディレクトリが見つかりません: $MODEL_PATH"
        echo "BERTモデルを先に訓練してください"
        exit 1
    fi

    # モデルファイルの確認
    if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
        echo "エラー: モデルファイルが見つかりません: $MODEL_PATH"
        echo "期待されるファイル: model.safetensors または pytorch_model.bin"
        exit 1
    fi
fi

# =============================================================================
# データ準備フェーズ
# =============================================================================

if [ "$SKIP_DATA_PREP" = false ]; then
    echo "=== データ準備フェーズ ==="
    echo "ProteinGymデータを準備中..."

    cd "$PROJECT_ROOT"

    # Pythonコマンド引数を準備
    DATA_PREP_ARGS=(
        "scripts/evaluation/bert/proteingym_data_preparation.py"
        "--output_dir" "$DATA_DIR"
        "--max_variants_per_assay" "$MAX_VARIANTS"
    )

    # ダウンロードフラグ
    if [ "$DOWNLOAD" = true ]; then
        DATA_PREP_ARGS+=("--download")
        echo "📥 ProteinGymデータをダウンロード中..."
    fi

    # サンプルのみフラグ
    if [ "$SAMPLE_ONLY" = true ]; then
        DATA_PREP_ARGS+=("--sample_only")
        echo "📝 サンプルデータのみ作成中..."
    fi

    python "${DATA_PREP_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "エラー: データ準備に失敗しました"
        exit 1
    fi

    echo "データ準備完了"
    echo ""

    # サンプルのみの場合はここで終了
    if [ "$SAMPLE_ONLY" = true ]; then
        echo "=== サンプルデータ作成完了 ==="
        echo "サンプルデータ: $DATA_DIR/bert_proteingym_sample.csv"
        echo ""
        echo "評価を実行するには、--sample_onlyオプションを外して再実行してください"
        exit 0
    fi
fi

# =============================================================================
# モデル評価フェーズ
# =============================================================================

if [ "$SKIP_EVALUATION" = false ]; then
    echo "=== モデル評価フェーズ ==="
    echo "BERT ProteinGym評価を実行中..."
    echo "モデル: $MODEL_PATH"
    echo "データ: $DATASET_PATH"

    # データセットの存在確認
    if [ ! -f "$DATASET_PATH" ]; then
        echo "エラー: データセットが見つかりません: $DATASET_PATH"
        echo "先にデータ準備を実行してください（--skip_data_prepを外す）"
        exit 1
    fi

    # Pythonコマンド引数を準備
    EVAL_ARGS=(
        "scripts/evaluation/bert/proteingym_evaluation.py"
        "--model_path" "$MODEL_PATH"
        "--proteingym_data" "$DATASET_PATH"
        "--output_dir" "$OUTPUT_DIR"
        "--device" "$DEVICE"
        "--batch_size" "$BATCH_SIZE"
    )

    # トークナイザーパスを追加（Noneでない場合）
    if [ "$TOKENIZER_PATH" != "None" ] && [ ! -z "$TOKENIZER_PATH" ]; then
        EVAL_ARGS+=("--tokenizer_path" "$TOKENIZER_PATH")
    fi

    python "${EVAL_ARGS[@]}"

    if [ $? -ne 0 ]; then
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

    # 最新の評価結果ディレクトリを探す
    LATEST_RESULT_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*bert_proteingym*" | sort | tail -1)

    if [ -z "$LATEST_RESULT_DIR" ]; then
        echo "エラー: 評価結果ディレクトリが見つかりません"
        echo "先に評価を実行してください（--skip_evaluationを外す）"
        exit 1
    fi

    echo "評価結果ディレクトリ: $LATEST_RESULT_DIR"

    # 結果ファイルの存在確認
    if [ ! -f "$LATEST_RESULT_DIR/bert_proteingym_results.json" ]; then
        echo "エラー: 評価結果ファイルが見つかりません"
        echo "期待されるファイル: $LATEST_RESULT_DIR/bert_proteingym_results.json"
        exit 1
    fi

    python "$PROJECT_ROOT/scripts/evaluation/bert/proteingym_visualization.py" \
        --results_dir "$LATEST_RESULT_DIR" \
        --output_dir "$LATEST_RESULT_DIR/visualizations"

    if [ $? -ne 0 ]; then
        echo "エラー: 可視化に失敗しました"
        exit 1
    fi

    echo "可視化完了"
    echo ""
fi

# =============================================================================
# 完了メッセージ
# =============================================================================

echo "=== BERT ProteinGym評価パイプライン完了 ==="
echo ""
echo "📁 出力ディレクトリ: $OUTPUT_DIR"

if [ "$SKIP_DATA_PREP" = false ]; then
    echo "📊 データ: $DATA_DIR"
fi

if [ "$SKIP_EVALUATION" = false ]; then
    echo "📈 評価結果: $LATEST_RESULT_DIR"
fi

if [ "$SKIP_VISUALIZATION" = false ]; then
    echo "📉 可視化: $LATEST_RESULT_DIR/visualizations"
fi

echo ""
echo "✅ 全ての処理が正常に完了しました"
