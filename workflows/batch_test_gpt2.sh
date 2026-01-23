#!/bin/bash

# GPT2チェックポイント一括テストスクリプト
# 使用方法: ./batch_test_gpt2.sh [search_directory]

set -e

SEARCH_DIR=${1:-"."}
OUTPUT_BASE_DIR="gpt2_test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=== GPT2チェックポイント 一括テストスクリプト ==="
echo "検索ディレクトリ: $SEARCH_DIR"
echo "タイムスタンプ: $TIMESTAMP"
echo ""

# 出力ディレクトリを作成
mkdir -p "${OUTPUT_BASE_DIR}_${TIMESTAMP}"

# Pythonの環境確認
echo "Python環境チェック中..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo ""

# チェックポイント検索
echo "チェックポイントを検索中..."
python gpt2/test_helper.py --search_dir="$SEARCH_DIR" --list_only

echo ""
echo "テストを開始しますか? (y/N): "
read -r CONFIRM

if [[ ! "$CONFIRM" =~ ^[yY]$ ]]; then
    echo "テストをキャンセルしました。"
    exit 0
fi

# 各ドメインのテストを実行
DOMAINS=("compounds" "molecule_nl" "genome" "protein_sequence" "rna")
SUCCESS_COUNT=0
TOTAL_COUNT=0

echo ""
echo "=== テスト実行開始 ==="

for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "[$domain] ドメインのチェックポイントを検索中..."
    
    # ドメイン特化のチェックポイントを検索
    CHECKPOINTS=$(find "$SEARCH_DIR" -type f -name "*.pt" -path "*${domain}*" 2>/dev/null || true)
    
    if [ -z "$CHECKPOINTS" ]; then
        echo "  チェックポイントが見つかりませんでした。"
        continue
    fi
    
    echo "$CHECKPOINTS" | while read -r checkpoint; do
        if [ -f "$checkpoint" ]; then
            TOTAL_COUNT=$((TOTAL_COUNT + 1))
            
            echo ""
            echo "  テスト中: $checkpoint"
            
            # テスト用の出力ディレクトリ
            CHECKPOINT_NAME=$(basename "$(dirname "$checkpoint")")
            TEST_OUTPUT_DIR="${OUTPUT_BASE_DIR}_${TIMESTAMP}/${domain}_${CHECKPOINT_NAME}"
            
            # テスト実行
            if python gpt2/test_checkpoint.py \
                --checkpoint_path="$checkpoint" \
                --domain="$domain" \
                --output_dir="$TEST_OUTPUT_DIR" \
                --max_test_samples=500 \
                --convert_to_hf; then
                
                echo "  ✓ テスト成功: $checkpoint"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                
                # 結果の要約を抽出
                if [ -f "${TEST_OUTPUT_DIR}/gpt2_test_report.json" ]; then
                    echo "  結果:"
                    python -c "
import json
try:
    with open('${TEST_OUTPUT_DIR}/gpt2_test_report.json', 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    print(f'    パープレキシティ: {results.get(\"perplexity\", \"N/A\"):.4f}' if isinstance(results.get('perplexity'), (int, float)) else '    パープレキシティ: N/A')
    print(f'    Top-1精度: {results.get(\"accuracy\", \"N/A\"):.4f}' if isinstance(results.get('accuracy'), (int, float)) else '    Top-1精度: N/A')
    print(f'    平均損失: {results.get(\"avg_loss\", \"N/A\"):.4f}' if isinstance(results.get('avg_loss'), (int, float)) else '    平均損失: N/A')
except Exception as e:
    print(f'    結果読み込みエラー: {e}')
"
                fi
            else
                echo "  ✗ テスト失敗: $checkpoint"
            fi
        fi
    done
done

echo ""
echo "=== テスト完了 ==="
echo "成功: $SUCCESS_COUNT / $TOTAL_COUNT"
echo "結果保存先: ${OUTPUT_BASE_DIR}_${TIMESTAMP}/"

# 統合レポートの生成
echo ""
echo "統合レポートを生成中..."

SUMMARY_FILE="${OUTPUT_BASE_DIR}_${TIMESTAMP}/test_summary.json"
python -c "
import json
import os
import glob
from pathlib import Path

base_dir = '${OUTPUT_BASE_DIR}_${TIMESTAMP}'
reports = glob.glob(os.path.join(base_dir, '**/gpt2_test_report.json'), recursive=True)

summary = {
    'timestamp': '${TIMESTAMP}',
    'total_tests': len(reports),
    'test_results': []
}

for report_path in reports:
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        domain = Path(report_path).parent.name.split('_')[0]
        checkpoint = data.get('checkpoint_path', 'Unknown')
        results = data.get('results', {})
        
        test_result = {
            'domain': domain,
            'checkpoint_path': checkpoint,
            'perplexity': results.get('perplexity'),
            'accuracy': results.get('accuracy'),
            'avg_loss': results.get('avg_loss'),
            'status': results.get('status', 'unknown')
        }
        summary['test_results'].append(test_result)
        
    except Exception as e:
        print(f'レポート読み込みエラー: {report_path}: {e}')

with open('${SUMMARY_FILE}', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f'統合レポートを保存: ${SUMMARY_FILE}')
"

echo ""
echo "全てのテストが完了しました！"
echo "詳細な結果は以下のディレクトリで確認できます:"
echo "  ${OUTPUT_BASE_DIR}_${TIMESTAMP}/"
