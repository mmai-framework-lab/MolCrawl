# 環境構築

以下のコマンドをそのまま実行すると **flash_attn**, **biopython** のインストールに失敗した。
そのため、これらを `environment.yaml` から削除して環境構築を行った。

```bash
git clone https://github.com/deskull-m/riken-dataset-fundational-model.git
cd riken-dataset-fundational-model
conda env create --name molcrawl --file=environment.yaml

pip install fcd
pip install guacamol
pip install deepchem
```

---

# GPT-2

## データの準備

### ダウンロード

[GuacaMol](https://figshare.com/projects/GuacaMol/56639) の train, valid, test をダウンロードして
`./benchmark/GuacaMol` に配置。

### データ前処理

* **Tokenizer の修正**
  `./src/compounds/utils/tokenizer.py` 内の `CompoundsTokenizer` において、
  `max_len` に応じた truncation と padding を行うように修正。

* **max_length の修正**
  GuacaMol の分子はトークン長が最大 102 のため、
  `./assets/configs/compounds.yaml` の `max_len` を **128** に変更。

* **Tokenization**
  `prepare_gpt2.py` を書き換え，packing処理を行わないように修正．これを用いて GuacaMol データセットを tokenization し、
  Hugging Face Datasets フォーマットに変換。
  出力先: `./benchmark/GuacaMol/compounds/training_ready_hf_dataset`

  ```bash
  python ./src/compounds/dataset/prepare_gpt2_.py ./assets/configs/compounds.yaml
  ```

---

## Config の修正

`./gpt2/configs/compounds/train_gpt2_config.py` 等を以下のように修正。

* `dataset_dir` ⇒ 前処理済み GuacaMol のパス
* `batch_size × gradient_accumulation = 256` となるよう調整
* `max_iter` ⇒ 10 エポック相当になるよう調整
* `eval_interval`, `eval_iters` ⇒ 1 エポックごとになるように設定
* `eos_token` ⇒ `sep_token=13` に変更
* `dataset_params` を追加
* `init_from = "fine-tuning"` を指定
* `checkpoint_path` ⇒ 事前学習モデルのパスを指定

---

## Fine-tuning

学習結果は `./benchmark/GuacaMol` 下に保存される。

```bash
CUDA_VISIBLE_DEVICES=0 python gpt2/train.py gpt2/configs/compounds/train_gpt2_config.py
CUDA_VISIBLE_DEVICES=0 python gpt2/train.py gpt2/configs/compounds/train_gpt2_medium_config.py
CUDA_VISIBLE_DEVICES=0 python gpt2/train.py gpt2/configs/compounds/train_gpt2_large_config.py
CUDA_VISIBLE_DEVICES=0 python gpt2/train.py gpt2/configs/compounds/train_gpt2_xl_config.py
```

---

## 分子生成

* **generate メソッド修正**
  `./gpt2/model.py` の `GPT.generate` を修正し、`eos_token` 出力時に生成を停止するよう変更。

* **初めのトークン出現頻度計算**
  `./benchmark/GuacaMol/guacamol_v1_train.smiles` 内で、分子の最初のトークンの出現頻度を算出。
  `./gpt2/sample_compound.py` にハードコーディングで埋め込み。

* **分子生成**
  `./benchmark/GuacaMol/small/` 等の下に `generated_compounds.txt` として 100,000 分子生成。

  ```bash
  CUDA_VISIBLE_DEVICES=0 python ./gpt2/sample_compound.py ./gpt2/configs/compounds/train_gpt2_config.py
  CUDA_VISIBLE_DEVICES=0 python ./gpt2/sample_compound.py ./gpt2/configs/compounds/train_gpt2_medium_config.py
  CUDA_VISIBLE_DEVICES=0 python ./gpt2/sample_compound.py ./gpt2/configs/compounds/train_gpt2_large_config.py
  CUDA_VISIBLE_DEVICES=0 python ./gpt2/sample_compound.py ./gpt2/configs/compounds/train_gpt2_xl_config.py
  ```

---

## 性能評価

ライブラリの内部コード`guacamol/utils/chemistry.py` を修正：

```diff
- from scipy import histgram
+ from numpy import histgram
```

評価は `./benchmark/GuacaMol/guacamol_evaluation.ipynb` にて実施。

---

# BERT

## Fine-tuning

事前学習あり/なしの BERT を 3 seed 分実験。
MoleculeNet ベンチマークで性能評価。
出力先:
`./benchmark/MoleculeNet/small/{pretrain_or_not}/{benchmark}/{seed}`

```bash
python ./bert/fine-tuning.py
```

---

## 性能評価

各実験で出力された `test_score.txt` からメトリックを読み取り、
3 seed 分の平均を計算。