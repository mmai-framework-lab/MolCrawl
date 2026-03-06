# BERT Checkpoint Tester (JA)

BERT チェックポイントの読み込み・推論・MLM・簡易評価を行うテスターの使い方です。

## 対象スクリプト

- `molcrawl/bert/test_checkpoint.py` (メイン)
- `molcrawl/bert/test_bert_checkpoint.sh` (補助スクリプト)

補助スクリプトは古いパス参照を含むため、**`python molcrawl/bert/test_checkpoint.py` の直接実行を推奨**します。

## 基本実行

```bash
python molcrawl/bert/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/compounds/bert-output/compounds-small/checkpoint-1000" \
  --domain compounds \
  --vocab_path assets/molecules/vocab.txt
```

`--checkpoint_path` は `from_pretrained` 可能なチェックポイントディレクトリを指定します。

## 主な引数

- `--checkpoint_path` (必須): テスト対象チェックポイント
- `--domain`: `compounds|molecule_nl|genome|protein_sequence|rna`
- `--vocab_path`: compounds/genome などで必要になる語彙/モデルファイル
- `--dataset_path`: 任意。指定時はデータセット評価も実施
- `--test_texts`: 任意。テスト用文字列を上書き

## 例

### Molecule NL

```bash
python molcrawl/bert/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/molecule_nl/bert-output/molecule_nl-small/checkpoint-1000" \
  --domain molecule_nl
```

### Genome

```bash
python molcrawl/bert/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/genome_sequence/bert-output/genome_sequence-small/checkpoint-1000" \
  --domain genome \
  --vocab_path "<LEARNING_SOURCE_DIR>/genome_sequence/spm_tokenizer.model"
```

## 出力

- `test_report.json` が `--checkpoint_path` の親ディレクトリに保存されます。

## トラブル時の確認

- `--checkpoint_path` が正しいか
- `--domain` が選択肢に含まれるか
- compounds/genome で `--vocab_path` が有効なファイルか
- 必要依存 (`torch`, `transformers`, `datasets`) があるか

## 参考

- サンプル語彙生成: `workflows/create_sample_vocab.sh`
