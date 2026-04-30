# GPT-2 Checkpoint Tester (JA)

GPT-2 チェックポイントの読み込み、HF 変換、パープレキシティ評価、生成テストを行う手順です。

## 対象スクリプト

- `molcrawl/models/gpt2/test_checkpoint.py` (メイン)
- `molcrawl/models/gpt2/test_helper.py` (チェックポイント探索ヘルパー)
- `workflows/batch_test_gpt2.sh` (一括テスト)

## 単体テスト実行

```bash
python molcrawl/models/gpt2/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/compounds/gpt2-output/compounds-small/ckpt.pt" \
  --domain compounds \
  --vocab_path assets/molecules/vocab.txt \
  --convert_to_hf \
  --output_dir gpt2_test_output
```

## 主な引数

- `--checkpoint_path` (必須): `.pt` チェックポイント
- `--output_dir`: 出力先（既定: `gpt2_test_output`）
- `--convert_to_hf`: HF 形式へ変換
- `--test_dataset_params`: JSON 文字列（例: `{"dataset_dir":"..."}`）
- `--domain`: `compounds|molecule_nat_lang|genome|protein_sequence|rna`
- `--vocab_path`: compounds などで語彙を指定
- `--max_test_samples`: 評価サンプル上限
- `--device`: 例 `cuda`, `cpu`, `cuda:0`

## 例

### Molecule NL

```bash
python molcrawl/models/gpt2/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/molecule_nat_lang/gpt2-output/molecule_nat_lang-small/ckpt.pt" \
  --domain molecule_nat_lang \
  --test_dataset_params '{"dataset_dir":"<LEARNING_SOURCE_DIR>/molecule_nat_lang/training_ready_hf_dataset"}' \
  --max_test_samples 1000
```

### Protein Sequence

```bash
python molcrawl/models/gpt2/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/protein_sequence/gpt2-output/protein_sequence-small/ckpt.pt" \
  --domain protein_sequence \
  --test_dataset_params '{"dataset_dir":"<LEARNING_SOURCE_DIR>/protein_sequence/training_ready_hf_dataset"}'
```

## ヘルパー/一括実行

```bash
# 検索のみ
python molcrawl/models/gpt2/test_helper.py --search_dir . --list_only

# 自動実行
python molcrawl/models/gpt2/test_helper.py --search_dir . --auto_run

# バッチ実行
bash workflows/batch_test_gpt2.sh
bash workflows/batch_test_gpt2.sh /path/to/checkpoints
```

## 出力

- `gpt2_test_report.json` (`--output_dir` 配下)
- `hf_model/` (`--convert_to_hf` 時)

## トラブル時の確認

- `--checkpoint_path` が `.pt` で有効か
- `--test_dataset_params` が JSON として正しいか
- compounds で `--vocab_path` が有効か
- `--device` が実行環境に合っているか
