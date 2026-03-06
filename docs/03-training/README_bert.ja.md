# Configuration Parameters — Molecule Natural Language Training

このドキュメントでは、README セクションのスタイルに合わせて、設定パラメータを説明します。

この設定は、分子自然言語記述データセットに対して BERT 系モデルを学習するためのものです。以下に各パラメータの説明を示します。

---

## 学習ハイパーパラメータ

- **`max_steps = 600000`**
  学習ステップ総数（オプティマイザ更新回数）。

- **`learning_rate = 6e-6`**
  オプティマイザの学習率。

- **`weight_decay = 1e-1`**
  過学習を抑えるための L2 正則化（weight decay）。

- **`log_interval = 100`**
  学習メトリクス（例: loss）を 100 ステップごとに記録。

---

## データセット・モデルパス

- **`dataset_dir`**
  前処理済み Hugging Face 互換データセットへのパス:
  これは処理済みデータセットへのパスです。詳細は [Training of GPT-2 model Section in](../01-getting_started/README.md) を参照してください。

- **`model_path = get_bert_output_path("molecule_nat_lang", model_size)`**
  モデル出力（チェックポイント、ログ）を保存するディレクトリ。

---

## トークナイズ設定

- **`max_length = 1024`**
  入力トークナイズ時の最大系列長。これを超える入力は切り詰められます。

---

## バッチ設定

- **`batch_size = 8`**
  GPU/デバイスごとの学習バッチサイズ。

- **`per_device_eval_batch_size = 1`**
  GPU/デバイスごとの評価バッチサイズ。

- **`gradient_accumulation_steps = 5 * 16`**
  オプティマイザ更新前に勾配を蓄積するステップ数。
  メモリ上限を超えずに、実効バッチサイズを大きくできます。

> **実効バッチサイズ** = `batch_size × gradient_accumulation_steps × num_GPUs`

---

## 特殊トークン

- **`start_instruction = 1`**
  命令開始を表すトークン ID。

- **`end_instruction = [518, 29914, 25580, 29962]`**
  命令ブロック終端を表すトークン ID 群。

- **`eos_token = 2`**
  シーケンス終端を表すトークン ID。

---

## モデルバリアント

- **`model_size = "small"`**
  使用するモデルバリアントを指定します。利用可能な計算資源や用途に応じて、`"small"`、`"medium"`、`"large"` から選択してください。
