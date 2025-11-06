#!/bin/bash

#
# GPT-2 Protein Classification評価パイプライン実行スクリプト
#
# このスクリプトは、GPT-2モデルを使用したタンパク質配列の分類評価を実行します。
# データ準備から評価、可視化までの全プロセスを自動で実行します。
#
# 注意: このスクリプトはbootstraps/ディレクトリから実行されることを想定しています
#
# 使用方法:
#   ./run_gpt2_protein_classification.sh [options]
#

set -e  # エラー時に停止

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

# デフォルト設定
MODEL_PATH="$PROJECT_ROOT/gpt2-output/protein_sequence-small/ckpt.pt"
DATA_PATH=""
TOKENIZER_PATH=""  # 空の場合は自動検出
OUTPUT_DIR="$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_protein_classification"
DATA_DIR="$LEARNING_SOURCE_DIR/protein_sequence/data/protein_classification"
CREATE_SAMPLE="false"
THRESHOLD="0.0"
SKIP_DATA_PREP=false
SKIP_EVALUATION=false
SKIP_VISUALIZATION=false

# ヘルプメッセージ
usage() {
    cat << EOF
GPT-2 Protein Classification評価パイプライン

使用方法: $0 [options]

オプション:
    -m, --model_path PATH       GPT-2モデルチェックポイントへのパス (デフォルト: gpt2-output/protein_sequence-small/ckpt.pt)
    -d, --data_path PATH        評価用データセットCSVへのパス
    --tokenizer PATH            トークナイザーパス（指定しない場合は自動検出）
    -o, --output_dir PATH       出力ディレクトリ (デフォルト: \$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_protein_classification)
    -s, --create_sample         サンプル評価データセットを作成
    -t, --threshold FLOAT       二値分類の閾値 (デフォルト: 0.0)
    --skip_data_prep            データ準備をスキップ
    --skip_evaluation           評価をスキップ
    --skip_visualization        可視化をスキップ
    -h, --help                  このヘルプメッセージを表示

例:
    # サンプルデータで評価（デフォルトモデル使用）
    $0 -s
    
    # カスタムデータセットで評価（デフォルトモデル使用）
    $0 -d my_protein_variants.csv
    
    # カスタムモデルとデータセットで評価
    $0 -m gpt2-output/protein_sequence-medium/ckpt.pt -d my_protein_variants.csv
    
    # 評価のみ実行（データ準備をスキップ）
    $0 -d data.csv --skip_data_prep

必要なCSVフォーマット:
    評価用データセットには以下のカラムが必要です:
    - sequence: タンパク質アミノ酸配列
    - variant_pos: バリアントの位置 (0-indexed)
    - ref_aa: 参照アミノ酸
    - alt_aa: 代替アミノ酸
    - pathogenic: 二値ラベル (1=病原性, 0=良性)

EOF
}

# Parse command line arguments
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
        --tokenizer)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--create_sample)
            CREATE_SAMPLE="true"
            shift
            ;;
        -t|--threshold)
            THRESHOLD="$2"
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
        --skip_visualization)
            SKIP_VISUALIZATION=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            usage
            exit 1
            ;;
    esac
done

# モデルファイルの存在確認（評価を実行する場合のみ）
if [ "$SKIP_EVALUATION" = false ]; then
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "エラー: GPT-2モデルチェックポイントが見つかりません: $MODEL_PATH"
        echo ""
        echo "利用可能なモデル:"
        ls -la "$PROJECT_ROOT/gpt2-output/" 2>/dev/null || echo "  gpt2-outputディレクトリが存在しません"
        exit 1
    fi
fi

# 設定表示
echo "=== GPT-2 Protein Classification評価パイプライン開始 ==="
echo "モデルパス: $MODEL_PATH"
echo "データパス: ${DATA_PATH:-サンプルデータを作成}"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "データディレクトリ: $DATA_DIR"
echo "サンプル作成: $CREATE_SAMPLE"
echo "閾値: $THRESHOLD"
echo ""

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p logs

# Python環境の確認
if ! command -v python &> /dev/null; then
    echo "エラー: Pythonが見つかりません"
    exit 1
fi

# 必要なPythonパッケージの確認（GPT-2評価用）
echo "Pythonパッケージを確認中..."
python -c "
import sys
required_packages = ['torch', 'sklearn', 'pandas', 'numpy', 'sentencepiece']
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
    if [[ "$CREATE_SAMPLE" == "true" ]] || [[ -z "$DATA_PATH" ]]; then
        echo "=== データ準備フェーズ ==="
        echo "サンプルデータを準備中..."
        
        cd "$PROJECT_ROOT"
        
        python scripts/evaluation/gpt2/protein_classification_data_preparation.py \
            --output_dir "$DATA_DIR" \
            --create_sample
        
        if [[ $? -ne 0 ]]; then
            echo "エラー: データ準備に失敗しました"
            exit 1
        fi
        
        # サンプルデータパスを設定
        if [[ -z "$DATA_PATH" ]]; then
            DATA_PATH="$DATA_DIR/protein_classification_sample.csv"
        fi
        
        echo "データ準備完了"
        echo ""
    fi
fi

# =============================================================================
# モデル評価フェーズ
# =============================================================================

if [ "$SKIP_EVALUATION" = false ]; then
    echo "=== モデル評価フェーズ ==="
    echo "GPT-2 Protein Classification評価を実行中..."
    echo "モデル: $MODEL_PATH"
    echo "データ: ${DATA_PATH:-サンプルデータを使用}"
    
    # データファイルの存在確認
    if [[ -n "$DATA_PATH" ]] && [[ ! -f "$DATA_PATH" ]]; then
        echo "エラー: 評価用データセットが見つかりません: $DATA_PATH"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Pythonコマンド引数を準備
    EVAL_ARGS=(
        "scripts/evaluation/gpt2/protein_classification_evaluation.py"
        "--model_path" "$MODEL_PATH"
        "--output_dir" "$OUTPUT_DIR"
        "--threshold" "$THRESHOLD"
    )
    
    if [[ -n "$DATA_PATH" ]]; then
        EVAL_ARGS+=("--data_path" "$DATA_PATH")
    fi
    
    # トークナイザーパスが指定されている場合は追加
    if [[ -n "$TOKENIZER_PATH" ]]; then
        EVAL_ARGS+=(--tokenizer_path "$TOKENIZER_PATH")
        echo "トークナイザー: $TOKENIZER_PATH"
    else
        echo "トークナイザー: 自動検出"
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
    
    # 結果ファイルの存在確認
    RESULTS_FILE="$OUTPUT_DIR/protein_classification_results.json"
    
    if [[ -f "$RESULTS_FILE" ]]; then
        python "$PROJECT_ROOT/scripts/evaluation/gpt2/protein_classification_visualization.py" \
            --results_file "$RESULTS_FILE" \
            --output_dir "$OUTPUT_DIR/visualizations"
        
        if [[ $? -eq 0 ]]; then
            echo "可視化完了"
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

echo "=== GPT-2 Protein Classification評価パイプライン完了 ==="
echo ""

if [[ -f "$OUTPUT_DIR/protein_classification_results.json" ]]; then
    echo "📋 評価メトリクス:"
    echo "======================"
    python -c "
import json
with open('$OUTPUT_DIR/protein_classification_results.json', 'r') as f:
    results = json.load(f)
    metrics = results.get('metrics', {})
    print(f'  • Accuracy:    {metrics.get(\"Accuracy\", 0):.4f}')
    print(f'  • Precision:   {metrics.get(\"Precision\", 0):.4f}')
    print(f'  • Recall:      {metrics.get(\"Recall\", 0):.4f}')
    print(f'  • F1-score:    {metrics.get(\"F1-score\", 0):.4f}')
    print(f'  • ROC-AUC:     {metrics.get(\"ROC-AUC\", 0):.4f}')
    print(f'  • PR-AUC:      {metrics.get(\"PR-AUC\", 0):.4f}')
    print(f'  • Sensitivity: {metrics.get(\"Sensitivity\", 0):.4f}')
    print(f'  • Specificity: {metrics.get(\"Specificity\", 0):.4f}')
" 2>/dev/null || echo "  メトリクスを読み込めませんでした"
    echo ""
fi

echo "📁 出力ディレクトリ: $OUTPUT_DIR"

if [ "$SKIP_DATA_PREP" = false ]; then
    echo "📊 データ: $DATA_DIR"
fi

echo ""
echo "✅ 全ての処理が正常に完了しました"