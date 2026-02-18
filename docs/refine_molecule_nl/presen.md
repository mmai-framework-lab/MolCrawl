# RIKEN Dataset Foundational Model プロジェクト

## 5つのAIモデル構築基盤の全体像

---

## 📋 プロジェクト概要

生物学・化学分野の大規模データセットを活用した5つの基盤モデル（Foundational Models）を構築・評価するための統合プラットフォーム

### 🎯 目的

- **マルチモーダル学習**: ゲノム、タンパク質、RNA、化合物、自然言語の5つのモダリティを統合
- **評価自動化**: 8種類の評価パイプラインによる性能検証
- **Webインターフェース**: データセット準備とモデル評価の進捗を可視化

---

## 🧬 5つのAIモデル

### 1. **Genome Sequence Model** (ゲノム配列)

- **アーキテクチャ**: BERT / GPT-2
- **データソース**: NCBI RefSeq
- **タスク**:
  - 病原性変異予測（ClinVar評価）
  - がん関連変異分類（COSMIC評価）
  - 遺伝病関連性評価（OMIM評価）
- **評価パイプライン**: 3種類

```
RefSeq Download → FASTA処理 → トークナイザー訓練 → Parquet変換
```

---

### 2. **Protein Sequence Model** (タンパク質配列)

- **アーキテクチャ**: BERT / GPT-2
- **データソース**: UniProt (Swiss-Prot + TrEMBL)
- **タスク**:
  - フィットネススコア予測（ProteinGym）
  - タンパク質分類
- **評価パイプライン**: 2種類

```
UniProt Download → FASTA抽出 → トークン化 → Parquet変換
```

---

### 3. **RNA Expression Model** (RNA発現データ)

- **アーキテクチャ**: BERT
- **データソース**: CellxGene (単一細胞RNA-seqデータ)
- **タスク**:
  - 細胞型分類
  - 遺伝子発現パターン解析
- **評価パイプライン**: 開発中

```
CellxGene API → データセットリスト構築 → H5AD/Loom変換 → トークン化 → Vocabulary構築
```

---

### 4. **Molecule Natural Language Model** (分子-自然言語)

- **アーキテクチャ**: BERT / GPT-2
- **データソース**: SMolInstruct (320万件の分子-テキストペア)
- **タスク**:
  - 分子構造からの特性予測
  - SMILES記法と自然言語の相互変換
  - 分子設計の自然言語指示
- **評価パイプライン**: 1種類

```
HuggingFace Hub Download → ZIP展開 → トークン化 → 統計処理
```

---

### 5. **Compounds Model** (化合物構造)

- **アーキテクチャ**: BERT / GPT-2
- **データソース**: OrganiX13 (PubChem, ChEMBL, ZINC)
- **タスク**:
  - 化合物特性予測
  - SMILES記法の学習
- **評価パイプライン**: 開発中

```
データセットDownload → SMILES Canonicalization → トークン化 → 統計分析
```

---

## 🏗️ システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                   Web Frontend (React.js)                │
│  - データセット準備進捗の可視化                            │
│  - 実験管理ダッシュボード                                  │
│  - ZINC化合物チェッカー                                    │
└────────────────────┬────────────────────────────────────┘
                     │ REST API
┌────────────────────┴────────────────────────────────────┐
│              Backend API Server (Node.js/Express)        │
│  - /api/dataset-progress                                 │
│  - /api/experiments                                       │
│  - /api/zinc-checker                                      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│              Python Processing Pipeline                  │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Data Preparation Scripts (5種類)                │   │
│  │  - preparation_script_genome_sequence.py         │   │
│  │  - preparation_script_protein_sequence.py        │   │
│  │  - preparation_script_rna.py                     │   │
│  │  - preparation_script_molecule_related_nat_lang │   │
│  │  - preparation_script_compounds.py               │   │
│  └─────────────────────────────────────────────────┘   │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Evaluation Pipelines (8種類)                     │   │
│  │  BERT: ClinVar, ProteinGym                       │   │
│  │  GPT-2: ClinVar, COSMIC, OMIM(Dummy/Real),      │   │
│  │         Protein Classification, ProteinGym       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Model Training (BERT / GPT-2)                    │   │
│  └─────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────┘
```

---

## 📊 データ処理フロー（統一パターン）

### Phase 1: データ準備

```python
1. Download/Fetch
   ↓
2. Format Conversion (FASTA→CSV, H5AD→Loom, etc.)
   ↓
3. Tokenization
   ↓
4. Parquet/Database変換
```

### Phase 2: モデル訓練

```python
1. Tokenizer構築
   ↓
2. Model初期化 (BERT/GPT-2)
   ↓
3. 学習ループ (DeepSpeed最適化)
   ↓
4. チェックポイント保存
```

### Phase 3: 評価

```python
1. Evaluation Script実行
   ↓
2. メトリクス計算 (Accuracy, AUC, Spearman等)
   ↓
3. 可視化生成 (10種類以上のグラフ)
   ↓
4. HTML/PDFレポート生成
```

---

## 🎨 可視化機能

### 各評価パイプラインが生成する可視化

1. **混同行列** (Confusion Matrix)
2. **ROC曲線** (ROC-AUC)
3. **Precision-Recall曲線**
4. **性能メトリクスの棒グラフ**
5. **スコア分布ヒストグラム**
6. **レーダーチャート** (全指標統合)
7. **散布図** (予測値 vs 実測値)
8. **サマリーダッシュボード**
9. **HTMLレポート**
10. **テキストサマリー**

---

## 🚀 実行例

### 1. データセット準備

```bash
# Genome Sequenceデータの準備
python scripts/preparation/preparation_script_genome_sequence.py \
    configs/genome_config.yaml

# Molecule NLデータのダウンロード
bash scripts/preparation/download_smolinstruct.sh
python scripts/preparation/preparation_script_molecule_related_nat_lang.py \
    configs/molecule_nl_config.yaml
```

### 2. 評価実行

```bash
# BERT ClinVar評価
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# GPT-2 ProteinGym評価
./workflows/run_gpt2_proteingym_evaluation.sh \
    -m gpt2-output/protein_sequence-large/ckpt.pt \
    -o results/proteingym
```

---

## 🔧 技術スタック

### バックエンド

- **Python 3.9+**
- **PyTorch** - モデル訓練
- **Transformers (HuggingFace)** - BERT/GPT-2実装
- **DeepSpeed** - 分散学習最適化
- **Datasets (HuggingFace)** - データセット管理

### フロントエンド

- **React.js** - UI構築
- **Node.js/Express** - APIサーバー
- **SQLite** - 実験履歴管理

### データ処理

- **Pandas** - データ操作
- **NumPy** - 数値計算
- **Biopython** - 生物学データ処理
- **RDKit** - 化合物処理

### 品質管理

- **Ruff** - Python linter
- **ESLint** - JavaScript linter
- **GitHub Actions** - CI/CD

---

## 📈 評価指標

### 分類タスク (ClinVar, COSMIC, Protein Classification)

- **Accuracy** - 全体精度
- **Precision/Recall/F1-Score** - クラス別性能
- **ROC-AUC** - 閾値非依存性能
- **PR-AUC** - 不均衡データ対応
- **Sensitivity/Specificity** - 医療応用向け

### 回帰タスク (ProteinGym)

- **Spearman相関** - 順位相関
- **MSE/RMSE** - 予測誤差
- **R²スコア** - 決定係数

---

## 🌟 主要機能

### 1. **柔軟な出力先指定**

全評価スクリプトで `-o` / `--output-dir` オプションをサポート

```bash
./run_bert_clinvar_evaluation.sh -o /custom/results/path
```

### 2. **GPU自動検出**

デフォルトでGPUを使用、CPUフォールバック対応

### 3. **デフォルトモデル**

小規模モデルがデフォルトで設定済み（テスト実行を高速化）

### 4. **進捗追跡**

Webインターフェースでリアルタイムに進捗確認

### 5. **エラーハンドリング**

- 行番号付きエラーメッセージ
- 詳細なログ出力
- 自動リトライ機構

---

## 📁 ディレクトリ構造

```
riken-dataset-fundational-model/
├── scripts/
│   ├── preparation/          # 5つのデータ準備スクリプト
│   └── evaluation/           # 8つの評価パイプライン
│       ├── bert/
│       └── gpt2/
├── workflows/               # 8つの評価実行スクリプト
│   ├── run_bert_clinvar_evaluation.sh
│   ├── run_bert_proteingym_evaluation.sh
│   ├── run_gpt2_clinvar_evaluation.sh
│   ├── run_gpt2_cosmic_evaluation.sh
│   ├── run_gpt2_omim_evaluation_dummy.sh
│   ├── run_gpt2_omim_evaluation_real.sh
│   ├── run_gpt2_protein_classification.sh
│   └── run_gpt2_proteingym_evaluation.sh
├── molcrawl-web/             # Webインターフェース
│   ├── src/
│   │   ├── App.js
│   │   ├── DatasetProgressCard.js
│   │   ├── ExperimentDashboard.js
│   │   └── ZincChecker.js
│   ├── api/
│   │   ├── dataset-progress.js
│   │   └── experiments.js
│   └── server.js
├── bert/                     # BERTモデル実装
├── gpt2/                     # GPT-2モデル実装
├── src/                      # 共通ユーティリティ
│   ├── config/
│   ├── utils/
│   └── molecule_related_nl/
└── learning_source_*/        # データセット保存先
```

---

## 🔄 CI/CD パイプライン

### GitHub Actions ワークフロー

1. **Python Linting (Ruff)**
   - すべてのPythonファイルをチェック
   - 重大なエラー（F, E9系）でビルド失敗

2. **JavaScript Linting (ESLint)**
   - molcrawl-web配下をチェック
   - エラー時にビルド失敗

3. **自動テスト** (将来実装予定)

---

## 📊 プロジェクトの成果

### コード品質

- **Ruff**: 165エラー → 8エラー (97%削減)
- **ESLint**: 34エラー → 0エラー (100%解消)

### 評価パイプライン

- **8種類の自動評価**
- **80種類以上の可視化グラフ**
- **HTMLレポート自動生成**

### ドキュメント

- **10個以上のREADME**
- **トラブルシューティングガイド**
- **使用例とベストプラクティス**

---

## 🚧 今後の展開

### 短期目標

1. RNA Expression Modelの評価パイプライン完成
2. Compounds Modelの評価パイプライン完成
3. データセット準備の完全自動化

### 中期目標

1. モデル間の転移学習
2. マルチモーダル統合モデル
3. リアルタイム推論API

### 長期目標

1. 創薬支援システムとの統合
2. 医療診断支援への応用
3. オープンソースコミュニティの構築

---

## 🎓 まとめ

### プロジェクトの強み

✅ **5つのモダリティ**を統一プラットフォームで扱える  
✅ **評価自動化**により再現性を確保  
✅ **Web UI**で非プログラマーも利用可能  
✅ **高品質コード**（Linter完全適合）  
✅ **詳細ドキュメント**で保守性確保

### 応用可能性

🧬 **創薬**: 化合物-タンパク質相互作用予測  
🏥 **医療**: 遺伝病リスク評価  
🔬 **基礎研究**: 遺伝子機能解析  
🤖 **AI研究**: マルチモーダル基盤モデル開発

---

## 📞 質疑応答

ご質問をお待ちしております！
