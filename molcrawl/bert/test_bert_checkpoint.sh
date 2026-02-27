#!/bin/bash

# BERTチェックポイントテスト実行スクリプト
# 使用方法: ./test_bert_checkpoint.sh <checkpoint_path> [domain]

set -e

# 引数チェック
if [ $# -lt 1 ]; then
    echo "使用方法: $0 <checkpoint_path> [domain]"
    echo "  checkpoint_path: テストするチェックポイントのパス"
    echo "  domain: compounds, genome, rna, molecule_nl のいずれか（オプション）"
    exit 1
fi

CHECKPOINT_PATH="$1"
DOMAIN="${2:-compounds}"

echo "=== BERTチェックポイントテスト開始 ==="
echo "チェックポイント: $CHECKPOINT_PATH"
echo "ドメイン: $DOMAIN"
echo "=================================="

# Python環境の確認
if ! command -v python &> /dev/null; then
    echo "Pythonが見つかりません。Python環境を確認してください。"
    exit 1
fi

# 必要なディレクトリの作成
mkdir -p bert/test_samples
mkdir -p bert/test_results

# テストサンプルの生成
echo "1. テストサンプルを生成中..."
python bert/generate_test_samples.py --domain "$DOMAIN" --output_dir bert/test_samples

# テストサンプルファイルの確認
SAMPLE_FILE="bert/test_samples/${DOMAIN}_test_samples.txt"
if [ ! -f "$SAMPLE_FILE" ]; then
    echo "テストサンプルファイルが見つかりません: $SAMPLE_FILE"
    exit 1
fi

# テストサンプルを読み込み
TEST_TEXTS=()
while IFS= read -r line; do
    TEST_TEXTS+=("$line")
done < "$SAMPLE_FILE"

echo "✓ ${#TEST_TEXTS[@]} 個のテストサンプルを読み込みました"

# 設定ファイルを読み込み
source src/config/env.sh

# データセットパスの推定
DATASET_PATH=""
case "$DOMAIN" in
    "compounds")
        DATASET_PATH="$COMPOUNDS_DATASET_DIR"
        ;;
    "genome")
        DATASET_PATH="outputs/genome_sequence/training_ready_hf_dataset"
        ;;
    "rna")
        DATASET_PATH="outputs/rna/training_ready_hf_dataset"
        ;;
    "molecule_nl")
        DATASET_PATH="outputs/molecule_related_natural_language/training_ready_hf_dataset"
        ;;
esac

echo "推定データセットパス: $DATASET_PATH"

# メインテストの実行
echo "2. チェックポイントテストを実行中..."

# テストコマンドの構築
TEST_CMD="python bert/test_checkpoint.py --checkpoint_path \"$CHECKPOINT_PATH\""

# データセットが存在する場合は追加
if [ -d "$DATASET_PATH" ]; then
    TEST_CMD="$TEST_CMD --dataset_path \"$DATASET_PATH\""
    echo "✓ データセットが見つかりました: $DATASET_PATH"
else
    echo "! データセットが見つかりません: $DATASET_PATH（スキップします）"
fi

# テストテキストを追加
for text in "${TEST_TEXTS[@]}"; do
    TEST_CMD="$TEST_CMD --test_texts \"$text\""
done

echo "実行コマンド: $TEST_CMD"
echo "3. テスト実行中..."

# テストを実行
eval "$TEST_CMD"

echo ""
echo "=== テスト完了 ==="

# 結果ファイルの確認
RESULT_FILE="$(dirname "$CHECKPOINT_PATH")/test_report.json"
if [ -f "$RESULT_FILE" ]; then
    echo "✓ テスト結果: $RESULT_FILE"

    # 結果のサマリーを表示
    if command -v jq &> /dev/null; then
        echo ""
        echo "=== テスト結果サマリー ==="
        jq '.results.status' "$RESULT_FILE" 2>/dev/null || echo "結果ファイルの解析に失敗しました"
    fi
else
    echo "! テスト結果ファイルが見つかりません"
fi

echo ""
echo "テスト完了。詳細な結果は上記のファイルを確認してください。"
