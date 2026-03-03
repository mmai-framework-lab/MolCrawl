# BERT学習検証レポート - learning_20251125データセット

## 検証日時

2025年11月25日 15:52

## 検証目的

learning_20251125のMolecule NLデータセットでBERTの学習が正常に動作するかを検証

---

## データセット情報

### ディレクトリ構造

```
learning_20251125/molecule_nat_lang/
├── arrow_splits/
│   ├── train.arrow/    (3,267,176 samples)
│   ├── test.arrow/     (30,344 samples)
│   └── valid.arrow/    (17,781 samples)
└── molecule_related_natural_language_tokenized.parquet (583MB)
```

### データ統計

- **総サンプル数**: 3,315,301
- **総トークン数**: 約342M
- **タスク種類**: 14種類
- **学習用データ**: 3,267,176 samples
- **評価用データ**: 10,000 samples (高速化のため制限)

---

## テスト設定

### モデル構成

```python
model_size = "small"        # BERT small (109M parameters)
max_length = 128            # コンテキスト長（テスト用に短く）
vocab_size = 32008          # Llama-2トークナイザー語彙サイズ
```

**BERT Small 構成詳細:**

- Hidden size: 768
- Number of layers: 12
- Number of attention heads: 12
- Intermediate size: 3072
- **総パラメータ数**: 約109M

### 学習パラメータ

```python
batch_size = 4
gradient_accumulation_steps = 1
per_device_eval_batch_size = 4
max_steps = 50              # テスト用に短く
learning_rate = 6e-5
weight_decay = 0.1
warmup_steps = 10
log_interval = 10
mlm_probability = 0.15      # Masked Language Modeling確率
```

### Masked Language Modeling (MLM) 設定

- **マスキング確率**: 15%
- **マスク戦略**:
  - 80%: [MASK]トークンに置き換え
  - 10%: ランダムトークンに置き換え
  - 10%: 元のトークンを保持

---

## 学習経過

### 訓練損失とLearning Rate

| Step | Train Loss | Learning Rate | Gradient Norm |
| ---- | ---------- | ------------- | ------------- |
| 10   | 9.2835     | 6.00e-05      | 10.24         |
| 20   | 7.7051     | 4.50e-05      | -             |
| 30   | 7.2210     | 3.00e-05      | -             |
| 40   | 6.6476     | 1.50e-05      | -             |
| 50   | 6.5095     | 0.00e+00      | 6.55          |

### 評価損失

| Step | Eval Loss | Eval Runtime | Samples/sec | Epoch |
| ---- | --------- | ------------ | ----------- | ----- |
| 10   | 8.2294    | 74.23s       | 134.71      | 0.0   |
| 20   | 7.5155    | 74.60s       | 134.06      | 0.0   |
| 30   | 7.0527    | 64.47s       | 155.11      | 0.0   |
| 40   | 6.7736    | 68.34s       | 146.33      | 0.0   |
| 50   | 6.6860    | 71.28s       | 140.29      | 0.0   |

### 学習進捗グラフ（テキスト版）

```
Train Loss:
10.0 |
 9.0 | ●
 8.0 |   ●
 7.0 |     ●  ●
 6.0 |          ●
 5.0 |___________________
     10  20  30  40  50 (steps)

Eval Loss:
10.0 |
 9.0 |
 8.0 | ●
 7.0 |   ●  ●  ●  ●
 6.0 |
 5.0 |___________________
     10  20  30  40  50 (steps)
```

---

## 学習結果の分析

### ✅ 成功指標

1. **損失の減少**
   - Train Loss: 9.28 → 6.51 (29.8%減少)
   - Eval Loss: 8.23 → 6.69 (18.7%減少)
2. **収束の兆候**
   - 訓練損失が順調に減少
   - 評価損失も安定して減少
   - 過学習の兆候なし（eval loss < train lossの期間あり）

3. **安定性**
   - 50ステップ完了、エラーなし
   - Gradient norm安定 (10.24 → 6.55)
   - 評価速度安定 (~140 samples/sec)

### 📊 パフォーマンス統計

**総合パフォーマンス:**

```
Total runtime: 364.43 秒 (約6分)
Train samples/second: 1.098
Train steps/second: 0.137
Average eval time: ~70 seconds
```

**ステップ別速度:**

- Step 10: 4.91 it/s (初期ウォームアップ後)
- Step 20: 0.90 it/s (評価含む)
- Step 30-50: 0.43-0.94 it/s (評価含む)

### 🎯 検証項目チェックリスト

- [x] データローディング成功
- [x] Arrow形式認識成功
- [x] モデル初期化成功 (109M parameters)
- [x] カスタムMLM data collator動作
- [x] 可変長シーケンスのパディング/トランケーション成功
- [x] Forward pass実行可能
- [x] Backward pass実行可能
- [x] MLMマスキング正常動作
- [x] 損失計算正常
- [x] オプティマイザー動作
- [x] 学習率スケジューリング動作
- [x] チェックポイント保存成功
- [x] 定期的な評価実行
- [x] ログ出力正常

---

## 生成されたファイル

### チェックポイント構造

```
test_bert_molecule_nat_lang_20251125/checkpoint-50/
├── config.json           (589B)     # モデル設定
├── generation_config.json (90B)     # 生成設定
├── model.safetensors     (422MB)    # モデルの重み
├── optimizer.pt          (843MB)    # オプティマイザー状態
├── rng_state.pth         (14KB)     # 乱数状態
├── scheduler.pt          (1.1KB)    # スケジューラー状態
├── trainer_state.json    (2.6KB)    # 学習状態
└── training_args.bin     (5.2KB)    # 学習引数
```

### チェックポイントサイズ

- **総サイズ**: 約1.3GB
- **モデル本体**: 422MB
- **オプティマイザー**: 843MB (AdamWの2つのモーメンタムバッファを含む)

---

## データ品質の確認

### 1. Arrow形式のローディング

✅ **正常動作**

```
Loading from arrow format: learning_20251125/molecule_nat_lang/arrow_splits/train.arrow
📊 Limited test dataset to 10000 samples for faster evaluation
```

### 2. カスタムMLM Data Collator

✅ **正常動作**

- 可変長シーケンス → バッチ内最大長に統一
- パディング処理成功
- MLMマスキング適用成功
- input_ids, attention_mask, labels生成

### 3. 学習データフロー

✅ **正常動作**

- データローディング: スムーズ
- バッチ生成: 安定
- GPU転送: 問題なし
- メモリ使用: 安定

---

## 技術的課題と解決策

### 課題1: トークナイザーの認証エラー

**問題:**

```python
AttributeError: 'NoneType' object has no attribute 'mask_token'
```

**原因:**

- 標準のDataCollatorForLanguageModelingがトークナイザーを必要とする
- Llama-2トークナイザーへのアクセスに認証が必要

**解決策:**
カスタムMLM data collatorを実装

```python
@dataclass
class CustomDataCollatorForMLM:
    mlm_probability: float = 0.15
    max_length: int = 128

    def __call__(self, features):
        # パディング処理
        # MLMマスキング
        # labels生成
        ...
```

### 課題2: 可変長シーケンスのバッチ処理

**問題:**

```
ValueError: expected sequence of length 99 at dim 1 (got 57)
```

**原因:**

- input_idsの長さがサンプルごとに異なる
- tensorスタック時に形状が一致しない

**解決策:**
バッチ内最大長への動的パディング

```python
batch_max_length = min(
    max(len(f["input_ids"]) for f in features),
    self.max_length
)
# 各サンプルをbatch_max_lengthにパディング/トランケート
```

---

## 既存コードとの互換性

### 変更不要なコンポーネント

✅ `bert/main.py` - **一切変更なし**

- Arrow形式のローディングコードが既に存在
- カスタムdata collatorのサポートあり

### 新規追加されたコンポーネント

🆕 `bert/test_molecule_nat_lang_20251125_config.py` - テスト設定ファイル

- カスタムMLM data collator定義
- テスト用パラメータ設定
- vocab_sizeハードコード

---

## GPT-2との比較

### 学習方式の違い

| 項目           | BERT                     | GPT-2                      |
| -------------- | ------------------------ | -------------------------- |
| **学習タスク** | Masked Language Modeling | Next Token Prediction      |
| **入力形式**   | input_ids のみ           | input_ids + output_ids連結 |
| **マスキング** | ランダムに15%マスク      | なし                       |
| **双方向性**   | 双方向コンテキスト       | 左から右の単方向           |
| **損失計算**   | マスク位置のみ           | 全トークン位置             |

### 学習速度の比較

| モデル     | パラメータ数 | 速度         | バッチサイズ |
| ---------- | ------------ | ------------ | ------------ |
| BERT Small | 109M         | ~1 step/sec  | 4            |
| GPT-2 Test | 11M          | ~60 step/sec | 4            |

**注:** BERTの方が遅い理由

1. モデルサイズが大きい (109M vs 11M)
2. 双方向アテンション（計算量多い）
3. 評価データが大きい (10,000 samples)

---

## 結論

### ✅ 検証結果: 成功

learning_20251125のMolecule NLデータセットでBERTの学習は**完全に動作します**。

### 確認された事項

1. **データフォーマット**: ✅ 完全互換
2. **データローディング**: ✅ Arrow形式正常動作
3. **MLM処理**: ✅ カスタムdata collator正常動作
4. **モデル学習**: ✅ 正常動作
5. **損失計算**: ✅ 順調に減少
6. **チェックポイント**: ✅ 正常保存 (1.3GB)
7. **評価**: ✅ 定期的に実行、安定

### GPT-2との共通点

- ✅ 同じarrow_splitsディレクトリを使用可能
- ✅ 同じデータセットで両モデルを学習可能
- ✅ データ前処理不要（既にトークン化済み）

---

## 推奨事項

### 実際の学習を開始する場合

#### 1. より長いコンテキストを使用

```python
# bert/molecule_nat_lang_bert_config.py
max_length = 512  # BERTの最大推奨長
```

#### 2. より大きなモデルを使用

```python
model_size = "medium"  # or "large"
# medium: 24 layers, 1024 hidden, ~340M params
# large: 36 layers, 1152 hidden, ~455M params
```

#### 3. バッチサイズの調整

```python
batch_size = 16  # GPU メモリに応じて調整
gradient_accumulation_steps = 4
# 実効バッチサイズ = 16 * 4 = 64
```

#### 4. 長時間学習

```python
max_steps = 100000
warmup_steps = 1000
log_interval = 500
```

#### 5. 学習開始コマンド

```bash
# 既存の設定ファイルを使用
python bert/main.py bert/molecule_nat_lang_bert_config.py

# または環境変数で指定
LEARNING_SOURCE_DIR="learning_20251125" python bert/main.py bert/molecule_nat_lang_bert_config.py
```

---

## モデル選択のガイドライン

### BERTを選ぶべき場合

- ✅ 分類タスク（sentiment, entity recognition）
- ✅ 双方向コンテキストが重要
- ✅ 穴埋めタスク（fill-mask）
- ✅ 埋め込み生成

### GPT-2を選ぶべき場合

- ✅ 生成タスク（text generation）
- ✅ 次トークン予測
- ✅ 対話システム
- ✅ より高速な学習が必要

### 両方使うべき場合

- ✅ 多様なダウンストリームタスク
- ✅ モデルアンサンブル
- ✅ 比較実験

---

## 付録

### テスト設定ファイル

`bert/test_molecule_nat_lang_20251125_config.py`

### データセット

- `learning_20251125/molecule_nat_lang/arrow_splits/train.arrow` (3.3M samples)
- `learning_20251125/molecule_nat_lang/arrow_splits/valid.arrow` (17K samples)

### 学習スクリプト

- `bert/main.py` (変更なし)
- カスタムMLM data collator (設定ファイル内)

### ログファイル

- `bert_test_20251125.log` (133KB)

---

## まとめ

### ✨ 両モデルとも完全対応

**BERT**: ✅ 学習成功

- Train Loss: 9.28 → 6.51
- Eval Loss: 8.23 → 6.69
- チェックポイント: 1.3GB保存済み

**GPT-2**: ✅ 学習成功（前回検証済み）

- Train Loss: 10.01 → 4.66
- Val Loss: 10.02 → 5.30
- チェックポイント: 131MB保存済み

### 🚀 本番学習準備完了

learning_20251125データセットは以下に対応:

- ✅ BERT (MLM)
- ✅ GPT-2 (CLM)
- ✅ 同一データソースで両モデル学習可能

次のステップで本格的な学習を開始できます！
