# BERTチェックポイント テストスクリプト

このディレクトリには、BERTモデルのチェックポイントを包括的にテストするためのスクリプトが含まれています。

## ファイル構成

- `test_checkpoint.py` - メインのテストスクリプト
- `test_bert_checkpoint.sh` - 自動テスト実行スクリプト
- `sample.py` - 単純なBERT推論サンプル

## 使用方法

### 1. 語彙ファイルの準備（compoundsドメインの場合）

```bash
# サンプル語彙ファイルを作成（実際のファイルがない場合）
./create_sample_vocab.sh
```

### 2. 基本的なテスト実行

```bash
# 化合物ドメインのモデルをテスト
python bert/test_checkpoint.py \
    --checkpoint_path "runs_train_bert_compounds/checkpoint-10000" \
    --domain compounds \
    --vocab_path "assets/molecules/vocab.txt"

# 分子関連自然言語ドメインのモデルをテスト
python bert/test_checkpoint.py \
    --checkpoint_path "runs_train_bert_molecule_nat_lang/checkpoint-10000" \
    --domain molecule_nat_lang

# ゲノム配列ドメインのモデルをテスト
python bert/test_checkpoint.py \
    --checkpoint_path "runs_train_bert_genome/checkpoint-10000" \
    --domain genome \
    --vocab_path "learning_source"
```

### 3. カスタムテストテキストでテスト

```bash
# DNA配列でテスト
python bert/test_checkpoint.py \
    --checkpoint_path "runs_train_bert_genome/checkpoint-10000" \
    --domain genome \
    --vocab_path "learning_source" \
    --test_texts "ATCGATCGATCGATCG" "GCTAGCTAGCTAGCTA" "AAATTTCCCGGGATCG"
```

### 4. データセット評価を含むテスト

```bash
python bert/test_checkpoint.py \
    --checkpoint_path "runs_train_bert_compounds/checkpoint-10000" \
    --domain compounds \
    --vocab_path "assets/molecules/vocab.txt" \
    --dataset_path "learning_source"
```

### 5. 自動テストスクリプトの使用

```bash
# 化合物ドメインの自動テスト
./bert/test_bert_checkpoint.sh "runs_train_bert_compounds/checkpoint-10000" compounds
```

## テスト内容

### 1. 基本機能テスト

- モデルとトークナイザーの読み込み確認
- 基本的な推論実行テスト

### 2. マスク言語モデリング（MLM）テスト

- マスクされたトークンの予測
- トップ5予測の表示

### 3. エンベディング生成テスト

- テキストのベクトル表現生成
- エンベディング間の類似度計算

### 4. バッチ処理テスト

- 異なるバッチサイズでの処理性能測定

### 5. パフォーマンステスト

- モデルサイズ、パラメータ数の確認
- GPU/CPUメモリ使用量の確認
- データセットでの損失・パープレキシティ計算

## 対応ドメイン

- `compounds` - 化合物（SMILES記法）
- `molecule_nat_lang` - 分子関連自然言語
- `genome` - ゲノム配列（SentencePieceトークナイザー）
- `protein_sequence` - タンパク質配列（ESMトークナイザー）
- `rna` - RNA配列（開発中）

## トラブルシューティング

### トークナイザーが読み込めない場合

1. **語彙ファイルが見つからない場合**:

   ```bash
   ./create_sample_vocab.sh
   ```

2. **ドメイン特化モジュールが見つからない場合**:
   - `src/`ディレクトリがプロジェクトルートに存在することを確認
   - 必要なPythonパッケージがインストールされていることを確認

3. **チェックポイントからトークナイザーが読み込めない場合**:
   - `--domain`パラメータを指定してドメイン特化トークナイザーを使用
   - `--vocab_path`でカスタム語彙ファイルを指定

### モデルが読み込めない場合

1. チェックポイントパスが正しいことを確認
2. 必要なライブラリ（transformers、torch）がインストールされていることを確認
3. GPU/CPUメモリが十分であることを確認

## 出力ファイル

- `test_report.json` - 詳細なテスト結果（チェックポイントディレクトリに保存）
- `assets/img/molecule_nat_lang_tokenized_*_lengths_dist.png` - トークン長分布グラフ

## 例

### Compounds（化合物）モデルのテスト例

```bash
# 1. 語彙ファイル作成
./create_sample_vocab.sh

# 2. テスト実行
python bert/test_checkpoint.py \
    --checkpoint_path "runs_train_bert_compounds/checkpoint-10000" \
    --domain compounds \
    --test_texts "CCO" "C1=CC=CC=C1" "CC(=O)O"
```

### Genome Sequence（ゲノム配列）モデルのテスト例

```bash
python bert/test_checkpoint.py \
    --checkpoint_path "runs_train_bert_genome/checkpoint-10000" \
    --domain genome \
    --vocab_path "learning_source" \
    --test_texts "ATCGATCGATCGATCGATCGATCGATCGATCG" "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
```
