#!/bin/bash

# BERT ClinVar評価スクリプト
# ClinVarデータベースを使用したBERT genome sequenceモデルの精度検証

set -e  # エラー時に停止

echo "🧬 Independent BERT Genome Sequence - ClinVar Evaluation"
echo "================================================================"
echo "🤖 BERT-based pathogenicity prediction (Independent Implementation)"
echo "📅 Date: $(date)"
echo "🚀 Using trained BERT model with safetensors"
echo

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# LEARNING_SOURCE_DIRの確認
if [ -z "$LEARNING_SOURCE_DIR" ]; then
    echo "エラー: LEARNING_SOURCE_DIR環境変数が設定されていません"
    echo "実行前に以下を設定してください:"
    echo "  export LEARNING_SOURCE_DIR=/path/to/learning_source"
    exit 1
fi

# 設定値
MODEL_PATH="$PROJECT_ROOT/runs_train_bert_genome_sequence/checkpoint-5000"  # 訓練済みBERTモデル（safetensors）
TOKENIZER_PATH="$LEARNING_SOURCE_DIR/genome_sequence/spm_tokenizer.model"  # SentencePieceトークナイザー
DATASET_PATH="$LEARNING_SOURCE_DIR/genome_sequence/data/clinvar/clinvar_evaluation_dataset.csv"
OUTPUT_DIR="$LEARNING_SOURCE_DIR/genome_sequence/report/bert_clinvar_evaluation"

# パラメータの確認
echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Tokenizer Path: $TOKENIZER_PATH"
echo "  Dataset Path: $DATASET_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo

# 前提条件チェック
check_requirements() {
    echo "Checking requirements..."
    
    # モデルファイルの存在確認
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: BERT model not found at $MODEL_PATH"
        echo "Please train the BERT model first or specify correct path"
        exit 1
    fi
    
    # トークナイザーファイルの存在確認
    if [ ! -f "$TOKENIZER_PATH" ]; then
        echo "Error: Tokenizer not found at $TOKENIZER_PATH"
        echo "Please prepare the SentencePiece tokenizer first"
        exit 1
    fi
    
    # データセットファイルの存在確認
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Error: ClinVar dataset not found at $DATASET_PATH"
        echo "Please ensure the 2000 ClinVar variants CSV file is in place"
        exit 1
    fi
    
    # データセットの行数確認
    TOTAL_VARIANTS=$(wc -l < "$DATASET_PATH")
    echo "📊 Dataset contains $((TOTAL_VARIANTS - 1)) variants"
    
    # Pythonパッケージの確認
    source miniconda/bin/activate conda 2>/dev/null || {
        echo "Error: Conda environment not available"
        echo "Please setup conda environment first"
        exit 1
    }
    
    python -c "import torch, transformers, sklearn, sentencepiece, pandas, numpy" 2>/dev/null || {
        echo "Error: Required Python packages not installed in conda environment"
        echo "Please install: torch, transformers, scikit-learn, sentencepiece, pandas, numpy"
        exit 1
    }
    
    echo "All requirements satisfied."
    echo
}

# メイン評価の実行
run_evaluation() {
    echo "Running BERT ClinVar evaluation..."
    echo "This may take several minutes depending on dataset size and model complexity."
    echo
    
    # 出力ディレクトリの作成
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$PROJECT_ROOT/logs"
    
    # GPU使用可能性の確認
    if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep "True"; then
        DEVICE="cuda"
        echo "Using GPU for evaluation"
    else
        DEVICE="cpu"
        echo "Using CPU for evaluation (this will be slower)"
    fi
    echo
    
    # BERT ClinVar評価の実行
    source miniconda/bin/activate conda
    python scripts/evaluation/bert/clinvar_evaluation.py \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --max_length 512 \
        $SAMPLE_OPTION \
        2>&1 | tee "$PROJECT_ROOT/logs/bert_clinvar_evaluation_$(date +%Y%m%d_%H%M%S).log"
    
    echo
    echo "BERT ClinVar evaluation completed!"
}

# 結果の要約表示
show_results() {
    echo "=== Evaluation Results Summary ==="
    
    if [ -f "$OUTPUT_DIR/bert_clinvar_evaluation_results.json" ]; then
        echo "Performance metrics:"
        source miniconda/bin/activate conda
        python -c "
import json
with open('$OUTPUT_DIR/bert_clinvar_evaluation_results.json', 'r') as f:
    metrics = json.load(f)
    print(f'  Accuracy: {metrics[\"accuracy\"]:.3f}')
    print(f'  Precision: {metrics[\"precision\"]:.3f}')
    print(f'  Recall: {metrics[\"recall\"]:.3f}')
    print(f'  F1-score: {metrics[\"f1_score\"]:.3f}')
    print(f'  AUC: {metrics[\"auc\"]:.3f}')
    print(f'  Total variants evaluated: {metrics[\"total_variants\"]}')
"
    else
        echo "Results file not found. Check for errors in the evaluation."
    fi
    
    echo
    echo "Output files generated:"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -la "$OUTPUT_DIR/"
    fi
    
    echo
    echo "Detailed results and visualizations saved to: $OUTPUT_DIR"
}

# BERT独自の結果分析
analyze_bert_results() {
    echo "🔍 BERT-Specific Analysis"
    echo "========================="
    
    if [ -f "$OUTPUT_DIR/bert_clinvar_evaluation_results.json" ]; then
        echo "🧠 BERT Model Insights:"
        source miniconda/bin/activate conda
        python -c "
import json
import pandas as pd

# BERT結果の読み込み
with open('$OUTPUT_DIR/bert_clinvar_evaluation_results.json', 'r') as f:
    metrics = json.load(f)

print('🎯 Performance Summary:')
print(f'   Accuracy: {metrics[\"accuracy\"]:.1%}')
print(f'   F1-Score: {metrics[\"f1_score\"]:.3f}')
print(f'   AUC-ROC:  {metrics[\"auc\"]:.3f}')
print()

print('🧬 Sequence Analysis:')
print(f'   MLM Score Difference: {metrics[\"mean_mlm_score_pathogenic\"] - metrics[\"mean_mlm_score_benign\"]:.3f}')
print(f'   (Pathogenic - Benign)')
print()

print('📊 Model Characteristics:')
print('   • Bidirectional context analysis')
print('   • Masked language modeling approach')
print('   • Sequence representation learning')
print('   • Independent of generative models')
print()

# パフォーマンス評価
if metrics['accuracy'] > 0.8:
    print('✅ Excellent pathogenicity prediction performance')
elif metrics['accuracy'] > 0.7:
    print('✅ Good pathogenicity prediction performance')
elif metrics['accuracy'] > 0.6:
    print('⚠️  Moderate pathogenicity prediction performance')
else:
    print('❌ Poor pathogenicity prediction performance - model may need retraining')
"
    else
        echo "❌ BERT results not found."
    fi
    echo
}

# クリーンアップ関数
cleanup() {
    echo "Cleaning up temporary files..."
    # 必要に応じて一時ファイルの削除
    echo "Cleanup completed."
}

# シグナルハンドラーの設定
trap cleanup EXIT

# オプション解析
SAMPLE_SIZE=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "🧬 Independent BERT ClinVar Evaluation"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sample-size N    Use only N samples for testing"
            echo "  --verbose         Enable verbose output"
            echo "  --help           Show this help message"
            echo ""
            echo "🤖 Features:"
            echo "  • Trained BERT genome sequence model"
            echo "  • Independent pathogenicity assessment"
            echo "  • MLM-based variant impact scoring"
            echo "  • Sequence representation analysis"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# サンプルサイズが指定された場合
if [ ! -z "$SAMPLE_SIZE" ]; then
    echo "Using sample size: $SAMPLE_SIZE"
    SAMPLE_OPTION="--sample_size $SAMPLE_SIZE"
else
    SAMPLE_OPTION=""
fi

# メイン実行フロー
main() {
    echo "🚀 Starting Independent BERT ClinVar Evaluation Pipeline..."
    echo
    
    check_requirements
    run_evaluation
    show_results
    analyze_bert_results
    
    echo "🎉 Independent BERT ClinVar Evaluation Completed Successfully!"
    echo "=============================================================="
    echo "📁 Results: $OUTPUT_DIR"
    echo "📋 Logs: logs/"
    echo "🧬 Model: Trained BERT Genome Sequence Model"
    echo "📊 Method: Independent pathogenicity assessment"
    echo
}

# スクリプトの実行
main "$@"