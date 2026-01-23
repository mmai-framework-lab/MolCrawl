#!/bin/bash

# GPT2チェックポイントテストの簡単な使用例デモンストレーション

echo "=== GPT2チェックポイントテスト 使用例デモ ==="
echo ""

# 基本的な使用例
echo "1. 基本的な単一チェックポイントテスト:"
echo "   python gpt2/test_checkpoint.py --checkpoint_path=out-compounds-small/ckpt.pt"
echo ""

echo "2. ドメイン指定付きテスト:"
echo "   python gpt2/test_checkpoint.py \\"
echo "       --checkpoint_path=out-compounds-small/ckpt.pt \\"
echo "       --domain=compounds \\"
echo "       --vocab_path=assets/molecules/vocab.txt"
echo ""

echo "3. Hugging Face変換付きテスト:"
echo "   python gpt2/test_checkpoint.py \\"
echo "       --checkpoint_path=out-compounds-small/ckpt.pt \\"
echo "       --domain=compounds \\"
echo "       --convert_to_hf \\"
echo "       --output_dir=converted_model"
echo ""

echo "4. 利用可能なチェックポイントの検索:"
echo "   python gpt2/test_helper.py --list_only"
echo ""

echo "5. 一括テスト実行:"
echo "   ./gpt2/batch_test_gpt2.sh"
echo ""

echo "=== 実際のテスト例 ==="

# 実際にcompoundsの小さなチェックポイントが存在する場合のテスト
if [ -f "out-compounds-small-6e-6wu200-6000-its/ckpt.pt" ]; then
    echo "✓ Compoundsチェックポイントが見つかりました。テストを実行します..."
    
    python gpt2/test_checkpoint.py \
        --checkpoint_path=out-compounds-small-6e-6wu200-6000-its/ckpt.pt \
        --domain=compounds \
        --vocab_path=assets/molecules/vocab.txt \
        --max_test_samples=10 \
        --output_dir=demo_test_results
    
    echo ""
    echo "✓ テスト完了！結果は demo_test_results/ で確認できます。"
    
    if [ -f "demo_test_results/gpt2_test_report.json" ]; then
        echo ""
        echo "=== テスト結果サマリー ==="
        python3 -c "
import json
try:
    with open('demo_test_results/gpt2_test_report.json', 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    stats = results.get('performance_stats', {})
    
    print(f'チェックポイント: {data.get(\"checkpoint_path\", \"N/A\")}')
    print(f'テスト実行時刻: {data.get(\"test_timestamp\", \"N/A\")}')
    print(f'総パラメータ数: {stats.get(\"total_params\", \"N/A\"):,}')
    print(f'語彙サイズ: {stats.get(\"vocab_size\", \"N/A\")}')
    print(f'Top-1精度: {results.get(\"accuracy\", \"N/A\"):.4f}' if isinstance(results.get('accuracy'), (int, float)) else 'Top-1精度: N/A')
    print(f'ステータス: {results.get(\"status\", \"N/A\")}')
    
    samples = results.get('generated_samples', [])
    if samples:
        print(f'生成サンプル数: {len(samples)}')
        print(f'最初のサンプル: {samples[0][:100]}...' if len(samples[0]) > 100 else f'最初のサンプル: {samples[0]}')
    
except Exception as e:
    print(f'結果読み込みエラー: {e}')
"
    fi
else
    echo "⚠ Compoundsチェックポイントが見つかりません。"
    echo "   利用可能なチェックポイントを確認中..."
    python gpt2/test_helper.py --list_only | head -10
fi

echo ""
echo "=== 追加情報 ==="
echo "詳細な使用方法は docs/README_test.md をご覧ください。"
echo "設定ファイルは gpt2/test_configs/ にあります。"
echo "バッチテストを実行するには: ./workflows/batch_test_gpt2.sh"
