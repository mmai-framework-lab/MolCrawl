#!/bin/bash
#"""
#ClinVar評価パイプライン実行スクリプト
#
#このスクリプトは、ClinVarデータの取得から評価、可視化までの
#全プロセスを自動で実行します。
#
#注意: このスクリプトはbootstraps/ディレクトリから実行されることを想定しています
#"""

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

# ログファイルの設定
mkdir -p "${LEARNING_SOURCE_DIR}/genome_sequence/logs/"
LOG_FILE="${LEARNING_SOURCE_DIR}/genome_sequence/logs/genome_sequence-clinvar-evaluation-$(date +%Y-%m-%d_%H-%M-%S).log"

# 全出力をログファイルと画面の両方に出力
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "ログファイル: ${LOG_FILE}"
echo ""

# デフォルト出力先（-o/--output-dirで上書き可能）
OUTPUT_DIR="$EVALUATION_OUTPUT_DIR/genome_sequence/report/clinvar_evaluation"
DATA_DIR="$EVALUATION_OUTPUT_DIR/genome_sequence/data/clinvar"  # データ準備時の出力先
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
ClinVar評価パイプライン

使用法: $0 [オプション]

オプション:
    -o, --output-dir PATH       出力ディレクトリ [default: \$EVALUATION_OUTPUT_DIR/genome_sequence/report/clinvar_evaluation]
    -m, --model-size SIZE       モデルサイズ (small/medium/large/xl) [default: small]
    -t, --tokenizer PATH        トークナイザーパス（指定しない場合は自動検出）
    -s, --sequence-length LEN   配列長 [default: 100]
    -n, --max-samples NUM       クラスあたりの最大サンプル数 [default: 1000]
    -b, --batch-size SIZE       バッチサイズ [default: 16]
    -d, --download              ClinVarデータをダウンロード
    -e, --eval-only             評価のみ実行（データ準備をスキップ）
    -v, --visualize-only        可視化のみ実行
    -h, --help                  このヘルプを表示

環境変数:
    LEARNING_SOURCE_DIR         入力データディレクトリ（読み取り専用）
    EVALUATION_OUTPUT_DIR       出力データディレクトリ（デフォルト: LEARNING_SOURCE_DIRと同じ）

例:
    # デフォルト出力先での実行
    $0 --download --model-size medium --max-samples 2000
    
    # カスタム出力ディレクトリを指定
    $0 --download -o /custom/output/clinvar_results
    
    # 環境変数で出力先を分離
    export EVALUATION_OUTPUT_DIR=/writable/output
    $0 --eval-only --model-size large
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
    
    python "$PROJECT_ROOT/scripts/evaluation/gpt2/clinvar_visualization.py" \
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
        echo "ClinVarデータをダウンロードしてバランスサンプリング中..."
        echo "陽性（病原性）1000件、陰性（良性）1000件をランダム抽出"
        
        # ステップ1: データセットから直接ランダム抽出（配列生成含む）
        # 参照ゲノムファイルのパスを設定
        REF_FASTA="$LEARNING_SOURCE_DIR/genome_sequence/data/GCA_000001405.28_GRCh38.p13_genomic.fna"
        
        if [[ ! -f "$REF_FASTA" ]]; then
            # .gzファイルを確認
            if [[ -f "$REF_FASTA.gz" ]]; then
                REF_FASTA="$REF_FASTA.gz"
                echo "参照ゲノム: $REF_FASTA (圧縮版)"
            else
                echo "警告: 参照ゲノムが見つかりません: $REF_FASTA"
                echo "HuggingFace Datasetsから直接取得します（配列生成なし）"
                REF_FASTA=""
            fi
        else
            echo "参照ゲノム: $REF_FASTA"
        fi
        
        # extract_random_clinvar_samples.pyを使用してバランスサンプリング
        if [[ -n "$REF_FASTA" ]]; then
            # 参照ゲノムがある場合: データセットから直接抽出して配列生成
            python "$PROJECT_ROOT/scripts/evaluation/gpt2/extract_random_clinvar_samples.py" \
                --ref_fasta "$REF_FASTA" \
                --output_csv "$DATA_DIR/clinvar_evaluation_dataset.csv" \
                --num_samples 2000 \
                --flank "$((SEQUENCE_LENGTH / 2))" \
                --seed 42
        else
            # 参照ゲノムがない場合: 既存の前処理スクリプトで取得後にサンプリング
            echo "従来の方法でデータ準備..."
            python "$PROJECT_ROOT/scripts/evaluation/gpt2/clinvar_data_preparation.py" \
                --download \
                --output_dir "$DATA_DIR" \
                --max_samples "$MAX_SAMPLES" \
                --sequence_length "$SEQUENCE_LENGTH"
            
            # 生成されたファイルからバランスサンプリング
            if [[ -f "$DATA_DIR/clinvar_evaluation_dataset.csv" ]]; then
                echo "バランスサンプリングを適用中..."
                python "$PROJECT_ROOT/scripts/evaluation/gpt2/extract_random_clinvar_samples.py" \
                    --input_csv "$DATA_DIR/clinvar_evaluation_dataset.csv" \
                    --output_csv "$DATA_DIR/clinvar_evaluation_dataset_balanced.csv" \
                    --num_samples 2000 \
                    --seed 42
                
                # バランス版を使用
                mv "$DATA_DIR/clinvar_evaluation_dataset_balanced.csv" "$DATA_DIR/clinvar_evaluation_dataset.csv"
            fi
        fi
    else
        # --downloadなしの場合は既存データを使用、存在しない場合は自動ダウンロード
        if [[ ! -f "$DATA_DIR/clinvar_evaluation_dataset.csv" ]]; then
            echo "ClinVarデータが存在しないため、自動的にダウンロードします..."
            
            # 参照ゲノムファイルのパスを設定
            REF_FASTA="$LEARNING_SOURCE_DIR/genome_sequence/data/GCA_000001405.28_GRCh38.p13_genomic.fna"
            
            if [[ ! -f "$REF_FASTA" ]]; then
                # .gzファイルを確認
                if [[ -f "$REF_FASTA.gz" ]]; then
                    REF_FASTA="$REF_FASTA.gz"
                    echo "参照ゲノム: $REF_FASTA (圧縮版)"
                else
                    echo "警告: 参照ゲノムが見つかりません: $REF_FASTA"
                    echo "HuggingFace Datasetsから直接取得します（配列生成なし）"
                    REF_FASTA=""
                fi
            else
                echo "参照ゲノム: $REF_FASTA"
            fi
            
            # extract_random_clinvar_samples.pyを使用してバランスサンプリング
            if [[ -n "$REF_FASTA" ]]; then
                # 参照ゲノムがある場合: データセットから直接抽出して配列生成
                python "$PROJECT_ROOT/scripts/evaluation/gpt2/extract_random_clinvar_samples.py" \
                    --ref_fasta "$REF_FASTA" \
                    --output_csv "$DATA_DIR/clinvar_evaluation_dataset.csv" \
                    --num_samples 2000 \
                    --flank "$((SEQUENCE_LENGTH / 2))" \
                    --seed 42
            else
                # 参照ゲノムがない場合: 既存の前処理スクリプトで取得後にサンプリング
                echo "従来の方法でデータ準備..."
                python "$PROJECT_ROOT/scripts/evaluation/gpt2/clinvar_data_preparation.py" \
                    --download \
                    --output_dir "$DATA_DIR" \
                    --max_samples "$MAX_SAMPLES" \
                    --sequence_length "$SEQUENCE_LENGTH"
                
                # 生成されたファイルからバランスサンプリング
                if [[ -f "$DATA_DIR/clinvar_evaluation_dataset.csv" ]]; then
                    echo "バランスサンプリングを適用中..."
                    python "$PROJECT_ROOT/scripts/evaluation/gpt2/extract_random_clinvar_samples.py" \
                        --input_csv "$DATA_DIR/clinvar_evaluation_dataset.csv" \
                        --output_csv "$DATA_DIR/clinvar_evaluation_dataset_balanced.csv" \
                        --num_samples 2000 \
                        --seed 42
                    
                    # バランス版を使用
                    mv "$DATA_DIR/clinvar_evaluation_dataset_balanced.csv" "$DATA_DIR/clinvar_evaluation_dataset.csv"
                fi
            fi
        else
            echo "既存のClinVarデータを使用: $DATA_DIR/clinvar_evaluation_dataset.csv"
        fi
    fi
    
    echo "データ準備完了"
    
    # データセットのバランス確認
    if [[ -f "$DATA_DIR/clinvar_evaluation_dataset.csv" ]]; then
        echo ""
        echo "データセット統計:"
        python -c "
import pandas as pd
df = pd.read_csv('$DATA_DIR/clinvar_evaluation_dataset.csv')
print(f'総サンプル数: {len(df)}')
if 'ClinicalSignificance' in df.columns:
    print('ClinicalSignificance分布:')
    print(df['ClinicalSignificance'].value_counts())
elif 'classification' in df.columns:
    print('Classification分布:')
    print(df['classification'].value_counts())
" 2>/dev/null || echo "統計表示に失敗しました"
    fi
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
    CLINVAR_DATA="$DATA_DIR/clinvar_evaluation_dataset.csv"
    if [[ ! -f "$CLINVAR_DATA" ]]; then
        # 代替パスを確認
        if [[ -f "$DATA_DIR/data/clinvar_evaluation_dataset.csv" ]]; then
            CLINVAR_DATA="$DATA_DIR/data/clinvar_evaluation_dataset.csv"
        else
            echo "エラー: ClinVarデータが見つかりません: $CLINVAR_DATA"
            echo "まずデータ準備を実行してください"
            exit 1
        fi
    fi
    
    echo "モデル評価を実行中..."
    echo "モデル: $MODEL_PATH"
    echo "データ: $CLINVAR_DATA"
    
    # Pythonコマンド引数を準備
    EVAL_ARGS=(
        "$PROJECT_ROOT/scripts/evaluation/gpt2/clinvar_evaluation.py"
        --model_path "$MODEL_PATH"
        --clinvar_data "$CLINVAR_DATA"
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

RESULTS_FILE="$OUTPUT_DIR/evaluation_results.json"
if [[ ! -f "$RESULTS_FILE" ]]; then
    echo "エラー: 評価結果ファイルが見つかりません: $RESULTS_FILE"
    exit 1
fi

python "$PROJECT_ROOT/scripts/evaluation/gpt2/clinvar_visualization.py" \
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
