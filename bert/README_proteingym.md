# BERT版ProteinGym評価システム

このドキュメントでは、BERT版のprotein_sequenceモデルを使用してProteinGymデータセットで評価を行う方法について説明します。

## 概要

BERT版ProteinGym評価システムは、訓練済みのBERTモデルを使用してタンパク質変異のフィットネススコアを予測し、実験的なDMS（Deep Mutational Scanning）スコアと比較評価を行います。

## 主要な特徴

- **独立実装**: GPT2システムから完全に分離された実装
- **BERT MLM**: マスク言語モデルによる双方向文脈理解
- **EsmSequenceTokenizer**: protein_sequence専用のトークナイザー使用
- **Safetensors対応**: 効率的なモデル読み込み
- **包括的評価**: 相関係数、MAE、RMSE等の詳細指標

## ファイル構成

```
bert/
├── proteingym_evaluation.py        # メイン評価スクリプト
└── configs/
    └── bert_proteingym_config.py   # 設定ファイル

run_bert_proteingym_evaluation.sh   # 実行スクリプト
```

## 使用方法

### 1. 基本的な評価実行

```bash
# ProteinGymデータセットで評価
./run_bert_proteingym_evaluation.sh --dataset /path/to/proteingym_data.csv

# サンプル数を指定
./run_bert_proteingym_evaluation.sh --dataset /path/to/proteingym_data.csv --sample_size 100
```

### 2. カスタム設定での実行

```bash
# モデルパスとバッチサイズを指定
./run_bert_proteingym_evaluation.sh \
    --dataset /path/to/proteingym_data.csv \
    --model_path runs_train_bert_protein_sequence/checkpoint-1000 \
    --batch_size 8 \
    --device cuda

# 出力ディレクトリを指定
./run_bert_proteingym_evaluation.sh \
    --dataset /path/to/proteingym_data.csv \
    --output_dir ./custom_results
```

### 3. テスト用サンプルデータの作成

```bash
# サンプルデータを作成
./run_bert_proteingym_evaluation.sh --create_sample_data --dataset ./sample_data.csv

# 作成されたサンプルデータで評価
./run_bert_proteingym_evaluation.sh --dataset ./sample_data.csv
```

## 必要な前提条件

### モデル
- 訓練済みBERTモデル: `runs_train_bert_protein_sequence/checkpoint-*`
- Safetensors形式またはPyTorch形式

### データ形式
ProteinGymデータは以下のカラムを含む必要があります：
- `mutated_sequence`: 変異後のタンパク質配列
- `DMS_score`: 実験的フィットネススコア
- `target_seq`: 野生型配列（オプション）
- `mutant`: 変異情報（オプション）

### 環境
- CUDA対応GPU（推奨）
- Conda環境: `conda activate conda`
- 必要パッケージ: torch, transformers, pandas, numpy, scipy, safetensors

## 出力結果

評価完了後、以下のファイルが生成されます：

```
bert_proteingym_evaluation_results/
├── bert_proteingym_results.json              # 主要結果（JSON）
├── bert_proteingym_detailed_results.csv      # 詳細結果（CSV）
└── bert_proteingym_evaluation_report.txt     # 評価レポート
```

### 主要評価指標

- **Spearman相関係数**: 順序相関（ランキング精度）
- **Pearson相関係数**: 線形相関
- **MAE**: 平均絶対誤差
- **RMSE**: 平均二乗誤差の平方根

### BERT固有の解析

- **MLMスコア**: マスク言語モデルによる配列尤度
- **フィットネススコア**: 変異型MLM - 野生型MLM
- **配列類似度**: [CLS]表現のコサイン類似度
- **病原性スコア**: フィットネスと類似度の統合スコア

## 実行例

```bash
# conda環境をアクティベート
source ./miniconda/etc/profile.d/conda.sh
conda activate conda

# サンプルデータでテスト
./run_bert_proteingym_evaluation.sh --create_sample_data --dataset ./test_data.csv
./run_bert_proteingym_evaluation.sh --dataset ./test_data.csv --sample_size 4

# 実際のProteinGymデータで評価
./run_bert_proteingym_evaluation.sh \
    --dataset /data/proteingym/DMS_substitutions.csv \
    --sample_size 1000 \
    --output_dir ./proteingym_results_1k
```

## パフォーマンス解釈

### 相関係数の目安
- **> 0.7**: 優秀な性能
- **0.5-0.7**: 良好な性能  
- **0.3-0.5**: 中程度の性能
- **< 0.3**: 限定的な性能

### BERTモデルの特徴
- **双方向注意**: 前後の文脈を同時に考慮
- **MLMアプローチ**: 直接的な配列確率評価
- **表現学習**: 深層学習による配列表現獲得
- **独立評価**: 生成モデルに依存しない評価

## トラブルシューティング

### よくある問題

1. **モデルが見つからない**
   ```bash
   # checkpoint-2000が存在することを確認
   ls runs_train_bert_protein_sequence/
   ```

2. **CUDA out of memory**
   ```bash
   # バッチサイズを減らす
   ./run_bert_proteingym_evaluation.sh --dataset data.csv --batch_size 4
   ```

3. **環境変数エラー**
   ```bash
   # LEARNING_SOURCE_DIRが自動設定されることを確認
   echo $LEARNING_SOURCE_DIR
   ```

## 開発者向け情報

### カスタマイズ
- `bert/proteingym_evaluation.py`: 評価ロジックのカスタマイズ
- `bert/configs/bert_proteingym_config.py`: デフォルト設定の変更
- `run_bert_proteingym_evaluation.sh`: 実行オプションの追加

### 拡張機能
- 可視化機能の追加
- 複数モデルの比較評価
- カスタム評価指標の実装

## 参考

このシステムは以下の実装を参考にしています：
- ClinVar評価システム（BERT版genome_sequence）
- GPT2版ProteinGym評価（scripts/proteingym_evaluation.py）

詳細な技術仕様については各ソースコードのコメントを参照してください。