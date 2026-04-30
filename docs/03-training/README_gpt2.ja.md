# GPT-2 テスト

このドキュメントでは、前処理済みデータセットの小さなサブセットを使って GPT-2 学習を検証する方法を説明します。
主な考え方は、小さな学習サブセットに意図的に過学習させることで、モデルがデータセットから学習できることを確認することです。

## 使い方

### 1. データセットのサブセットを準備

実行:

`python gpt2/configs/<dataset>/prepare.py path/to/the/tokenized/dataset`

このステップでは:

- データセットを読み込み、
- サブセットをサンプリングし、
- 同一長のバッチを作成します。

サブセットサイズは以下で制御します:

- `--training-set-subset-len`
- `--test-set-subset-len`

ルール:

- 値が `< 1` の場合: 全データに対する割合として解釈
- 値が `>= 1` の場合: サンプル数として解釈（`1` は 1 サンプル）

### 2. GPT-2 を学習

実行:

`python gpt2/train.py path/to/corresponding/dataset/train_gpt2_config.py`

各データセットには学習パラメータを定義した `train_gpt2_config.py` が対応しています。

例:

`python gpt2/train.py gpt2/configs/molecule_nat_lang/train_gpt2_large_config.py`

これで学習が開始され、チェックポイント（例: `out/ckpt.pt`）が保存されます。

### 3. Multi-GPU / DDP（任意）

`torchrun` が利用可能な場合、複数 GPU で実行できます。

単一ノード・4GPU の例:

`torchrun --standalone --nproc_per_node=4 config_file.py`

### 4. 学習済みチェックポイントからサンプル生成

実行:

`python gpt2/sample.py {config.py}`

学習時に使用したものと同じ config ファイルを使ってください。

例:

`python gpt2/sample.py gpt2/configs/molecule_nat_lang/train_gpt2_large_config.py`

## 設定パラメータ

以下で設定します:

`path/to/corresponding/dataset/train_gpt2_config.py`

### 重要パラメータ

#### `tokenizer_path`

一部データセットではこのパラメータが必要です。その場合、データ前処理で生成された tokenizer を指定してください。

- Genome Sequence:
  `molcrawl/data/genome_sequence/preparation.py` で生成される `spm_tokenizer.model` を使用
  （`assets/configs/genome_sequence.yaml` の `output_dir` 配下）
- Compounds:
  `molcrawl/data/compounds/preparation.py` で生成される `vocab.txt` を使用
  （`assets/configs/compounds.yaml` の `vocab_path`）
- Molecule Natural Language:
  設定不要（`MoleculeNatLangTokenizer` が config 内で初期化される）
- Protein Sequence:
  設定不要（`EsmSequenceTokenizer` が config 内で初期化される）

#### `dataset_dir`

処理済みデータセットへのパス（[使い方](#使い方) のステップ1の出力）。
前処理スクリプトが出力したデータセットパスを設定してください。

### ログ

- `tensorboard`:
  TensorBoard のメトリクス記録（例: loss、learning rate）を有効化
- `tensorboard_dir`:
  TensorBoard ログの保存先ディレクトリ

### 出力

- `out_dir`:
  チェックポイントとログの出力先ディレクトリ

### バッチ設定

- `batch_size`:
  GPU ごとの 1 バッチあたりサンプル数
- `block_size`:
  最大トークン系列長
- `gradient_accumulation_steps`:
  オプティマイザ更新前の勾配蓄積ステップ数

実効バッチサイズ:

`batch_size × block_size × gradient_accumulation_steps × num_GPUs`

例:

`8 × 1024 × 80 × 8 = 524,288 tokens`

### 学習率スケジュール

- `max_iters`:
  総学習イテレーション数
- `lr_decay_iters`:
  `learning_rate` から `min_lr` まで減衰させるイテレーション数
- `warmup_iters`:
  0 から `learning_rate` までウォームアップするイテレーション数
- `learning_rate`:
  ウォームアップ後の最大学習率
- `min_lr`:
  減衰終了時の最小学習率

### 評価とログ

- `eval_interval`:
  評価頻度（イテレーション単位）
- `eval_iters`:
  評価に使うバッチ数
- `log_interval`:
  ログ出力頻度（イテレーション単位）

### 正則化

- `weight_decay`:
  L2 正則化の強さ
