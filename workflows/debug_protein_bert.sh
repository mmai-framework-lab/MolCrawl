#!/bin/bash

# Protein sequence BERT学習のトラブルシューティングスクリプト
# 使用方法: ./debug_protein_bert.sh

echo "=== Protein Sequence BERT学習 デバッグ ==="

# 設定ファイルを読み込み
source molcrawl/core/env.sh

# 1. データセットの確認
DATASET_DIR="$UNIPROT_DATASET_DIR"

if [ -d "$DATASET_DIR" ]; then
    echo "✓ データセットディレクトリが見つかりました: $DATASET_DIR"

    # データセットの構造を確認
    echo "データセット内容:"
    ls -la "$DATASET_DIR"

    # データセットの最初の数サンプルを確認するPythonスクリプトを実行
    python3 << 'EOF'
import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from datasets import load_from_disk
    from config.paths import UNIPROT_DATASET_DIR

    dataset = load_from_disk(UNIPROT_DATASET_DIR)

    print(f"✓ データセット読み込み成功")
    print(f"データセットの分割: {list(dataset.keys())}")

    if "train" in dataset:
        train_dataset = dataset["train"]
        print(f"訓練データ数: {len(train_dataset)}")
        print(f"データのカラム: {train_dataset.column_names}")

        # 最初のサンプルを確認
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"サンプルデータ: {sample}")
            print(f"サンプルデータのキー: {list(sample.keys())}")

            # 'text'または'sequence'キーをチェック
            if 'text' in sample:
                print(f"テキストサンプル (最初の100文字): {sample['text'][:100]}")
            elif 'sequence' in sample:
                print(f"配列サンプル (最初の100文字): {sample['sequence'][:100]}")

except Exception as e:
    print(f"✗ データセット読み込みエラー: {e}")
EOF

else
    echo "✗ データセットディレクトリが見つかりません: $DATASET_DIR"
    echo "利用可能なディレクトリ:"
    find . -name "*uniprot*" -type d 2>/dev/null || echo "uniprotディレクトリが見つかりません"
    find . -name "*protein*" -type d 2>/dev/null || echo "proteinディレクトリが見つかりません"
fi

echo ""
echo "=== トークナイザーテスト ==="

# 2. トークナイザーのテスト
python3 << 'EOF'
import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer

    print("✓ BERT互換プロテイントークナイザーのインポート成功")

    # トークナイザーを作成
    tokenizer = create_bert_protein_tokenizer()
    print(f"✓ トークナイザー作成成功")
    print(f"語彙サイズ: {len(tokenizer.get_vocab())}")
    print(f"モデル入力名: {tokenizer.model_input_names}")

    # サンプル配列でテスト
    test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

    # トークナイズテスト
    result = tokenizer(test_sequence, return_tensors="pt", padding=True, truncation=True)
    print(f"✓ トークナイズ成功")
    print(f"出力キー: {list(result.keys())}")
    print(f"input_ids形状: {result['input_ids'].shape}")

    if 'sequence_tokens' in result:
        print("✗ まだsequence_tokensキーが残っています")
    else:
        print("✓ input_idsキーが正しく使用されています")

except Exception as e:
    print(f"✗ トークナイザーテストエラー: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo "=== 設定ファイルの確認 ==="

CONFIG_FILE="molcrawl/tasks/pretrain/configs/protein_sequence/bert_small.py"
if [ -f "$CONFIG_FILE" ]; then
    echo "✓ 設定ファイルが見つかりました: $CONFIG_FILE"
    echo "設定ファイル内容の抜粋:"
    head -20 "$CONFIG_FILE"
else
    echo "✗ 設定ファイルが見つかりません: $CONFIG_FILE"
fi

echo ""
echo "=== 推奨される修正手順 ==="
echo "1. データセットの 'input_ids' カラム名を確認"
echo "2. BERT互換トークナイザーの動作を確認"
echo "3. 必要に応じてデータ前処理スクリプトを修正"
echo "4. 修正後、BERT学習を再実行"

echo ""
echo "=== デバッグ完了 ==="
