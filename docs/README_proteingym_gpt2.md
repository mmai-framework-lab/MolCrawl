# ProteinGym Evaluation for Protein Sequence Models

このディレクトリには、protein_sequenceのGPT2モデルをProteinGymデータベースで評価するためのスクリプト群が含まれています。

## ProteinGymデータセット

ProteinGymは、タンパク質の変異効果予測とタンパク質設計のための大規模ベンチマークデータセットです（[論文](https://pubmed.ncbi.nlm.nih.gov/38106144/)）。このプロジェクトでは、訓練済みのGPT-2 protein sequenceモデルを使用して、ProteinGymの深度変異スキャニング（DMS）アッセイデータで適合性予測の精度を評価します。

### 利用可能なデータセット

**推奨データセット（protein_sequence評価用）:**

- **DMS Substitutions**: 深度変異スキャニングの単一アミノ酸置換データ（メイン評価用）
- **DMS Reference**: アッセイのメタデータと詳細情報
- **Clinical Substitutions**: 臨床的意義のある変異データ（補完評価用）
- **Clinical Reference**: 臨床変異のメタデータ

**追加データセット:**

- **DMS Indels**: 挿入・欠失変異データ
- **MSA Files**: 多重配列アライメント（高度な分析用）
- **Protein Structures**: AlphaFold2予測構造（構造ベース分析用）

データは[ProteinGym公式サイト](https://proteingym.org/download)から取得されます（v1.3）。

## ファイル構成

### メインスクリプト

- `scripts/proteingym_evaluation.py` - ProteinGym評価のメインスクリプト
- `scripts/proteingym_data_preparation.py` - ProteinGymデータのダウンロードと前処理
- `scripts/proteingym_visualization.py` - 評価結果の可視化
- `run_proteingym_evaluation.sh` - 評価実行用のシェルスクリプト

### 設定ファイル

- `README_proteingym.md` - このファイル

## 必要な環境

### Pythonパッケージ

```bash
pip install torch numpy pandas scikit-learn sentencepiece scipy matplotlib seaborn requests tqdm biopython
```

### システム要件

- Python 3.8以上
- CUDA対応GPU（推奨、CPUでも実行可能）
- 十分なメモリ（評価データのサイズによる）

## 使用方法

### 1. 基本的な評価

```bash
# 訓練済みモデルでProteinGymデータを評価
./run_proteingym_evaluation.sh \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --data_path proteingym_data/YOUR_ASSAY.csv
```

### 2. サンプルデータでのテスト

```bash
# サンプルデータを作成して評価
./run_proteingym_evaluation.sh \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --create-sample \
    --data_path sample_data.csv \
    --visualize
```

### 3. ProteinGymデータの自動ダウンロード

```bash
# 推奨データセットを自動ダウンロードして評価
./run_proteingym_evaluation.sh \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --download-data \
    --visualize
```

### 4. 個別スクリプトの実行

#### データ準備

```bash
# 推奨データセットをダウンロード（protein_sequence評価に最適）
python scripts/proteingym_data_preparation.py --download recommended --data_dir proteingym_data/

# 利用可能なアッセイを一覧表示
python scripts/proteingym_data_preparation.py --list_assays --data_type substitutions

# テスト用の小さなアッセイを取得
python scripts/proteingym_data_preparation.py --get_test_assays 5

# 特定のアッセイデータを準備
python scripts/proteingym_data_preparation.py --prepare_assay ASSAY_ID --max_variants 1000

# 個別データセットのダウンロード
python scripts/proteingym_data_preparation.py --download substitutions --data_dir proteingym_data/
python scripts/proteingym_data_preparation.py --download clinical_substitutions --data_dir proteingym_data/
```

#### 評価実行

```bash
# メイン評価
python scripts/proteingym_evaluation.py \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --proteingym_data proteingym_data/ASSAY_ID.csv \
    --output_dir results/ \
    --batch_size 32 \
    --device cuda
```

#### 可視化

```bash
# 評価結果の可視化
python scripts/proteingym_visualization.py \
    --results_file results/evaluation_results.json \
    --output_dir results/visualizations/
```

## 評価指標

このプロジェクトでは以下の指標を使用してモデルの性能を評価します：

### 主要指標

- **Spearman相関係数**: 予測値と真の値の順位相関
- **Pearson相関係数**: 予測値と真の値の線形相関
- **MAE (Mean Absolute Error)**: 平均絶対誤差
- **RMSE (Root Mean Square Error)**: 平均二乗誤差の平方根

### 解釈

- **相関係数**: 1.0に近いほど良い予測性能
- **MAE/RMSE**: 0に近いほど良い予測性能
- ProteinGymでは通常、Spearman相関係数が主要な評価指標として使用されます

## 出力ファイル

評価実行後、以下のファイルが出力ディレクトリに生成されます：

```text
output_dir/
├── evaluation_results.json          # 詳細な評価結果（JSON形式）
├── evaluation_report.txt           # 人間が読みやすい評価レポート
└── visualizations/                 # 可視化ファイル（--visualizeオプション使用時）
    ├── correlation_scatter.png     # 相関散布図
    ├── score_distributions.png     # スコア分布比較
    ├── residuals.png              # 残差プロット
    ├── performance_metrics.png     # パフォーマンス指標
    └── mutation_analysis.png       # 変異タイプ別分析
```

## スクリプト詳細

### proteingym_evaluation.py

ProteinGym評価のメインスクリプトです。

**主要機能:**

- 訓練済みGPT-2モデルの読み込み
- ProteinGymデータの前処理
- 野生型と変異型の尤度比較によるフィットネススコア計算
- 相関係数、MAE、RMSEなどの評価指標計算

**オプション:**

```text
--model_path PATH           # 訓練済みモデルのパス（必須）
--proteingym_data PATH      # ProteinGymデータファイル（必須）
--output_dir PATH           # 出力ディレクトリ
--batch_size INT            # バッチサイズ
--device STR                # 実行デバイス（cuda/cpu）
--tokenizer_path PATH       # トークナイザーパス（自動検出）
--create_sample_data        # サンプルデータ作成フラグ
```

### proteingym_data_preparation.py

ProteinGymデータのダウンロードと前処理を行います。

**主要機能:**

- ProteinGymデータセットの自動ダウンロード
- ZIPファイルの展開
- 特定アッセイデータの抽出と前処理
- テスト用サンプルデータの生成

**オプション:**

```text
--data_dir PATH             # データ保存ディレクトリ
--download STR              # ダウンロードタイプ（substitutions/indels/reference/all）
--prepare_assay STR         # 準備するアッセイID
--max_variants INT          # 最大変異数
--create_test PATH          # テストデータ作成
--list_assays               # 利用可能アッセイの一覧表示
```

### proteingym_visualization.py

評価結果の可視化を行います。

**主要機能:**

- 相関散布図の生成
- スコア分布の比較
- 残差分析
- パフォーマンス指標の棒グラフ
- 変異タイプ別分析

**オプション:**

```text
--results_file PATH         # 評価結果JSONファイル（必須）
--prediction_file PATH      # 予測データCSVファイル（オプション）
--output_dir PATH           # 可視化出力ディレクトリ
--format STR                # 出力形式（png/pdf/svg）
```

## データ形式

### 入力データ形式（ProteinGym）

ProteinGymデータは以下のカラムを含むCSVファイルです：

```csv
mutant,mutated_sequence,DMS_score
A1V,VLKGDLSGLTQVKSGQDKGLT...,0.85
L2P,APKGDLSGLTQVKSGQDKGLT...,0.15
WT,ALKGDLSGLTQVKSGQDKGLT...,1.0
```

**必須カラム:**

- `mutant`: 変異記述（例: "A1V", "WT"）
- `mutated_sequence`: 変異後のアミノ酸配列
- `DMS_score`: Deep Mutational Scanning実験スコア

**オプションカラム:**

- `target_seq`: 野生型配列（ない場合は自動推定）

### 出力データ形式

#### evaluation_results.json

```json
{
  "spearman_correlation": 0.75,
  "spearman_p_value": 1e-10,
  "pearson_correlation": 0.7,
  "pearson_p_value": 1e-8,
  "mae": 0.15,
  "rmse": 0.2,
  "n_variants": 500
}
```

## トラブルシューティング

### よくある問題

1. **CUDA out of memory エラー**

   ```bash
   # バッチサイズを小さくする
   --batch_size 8

   # CPUを使用する
   --device cpu
   ```

2. **トークナイザーが見つからない**

   ```bash
   # 明示的にパスを指定
   --tokenizer_path path/to/spm_tokenizer.model
   ```

3. **データファイルが見つからない**

   ```bash
   # データを自動ダウンロード
   --download-data

   # またはサンプルデータを作成
   --create-sample
   ```

### ログ確認

詳細なログは `logs/` ディレクトリに保存されます：

```bash
tail -f logs/proteingym_evaluation_YYYYMMDD_HHMMSS.log
```

## ベンチマーク例

### 期待される性能範囲

ProteinGymでの典型的な性能指標：

| モデルタイプ   | Spearman相関 | 備考         |
| -------------- | ------------ | ------------ |
| ランダム       | ~0.0         | ベースライン |
| 進化情報ベース | 0.3-0.6      | MSA利用      |
| 言語モデル     | 0.4-0.7      | Large scale  |
| 構造情報利用   | 0.5-0.8      | 最高性能     |

### ベンチマーク実行例

```bash
# 小規模テスト（5分程度）
./run_proteingym_evaluation.sh \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --create-sample \
    --data_path test_100.csv \
    --batch_size 8

# 中規模評価（30分程度）
./run_proteingym_evaluation.sh \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --data_path specific_assay.csv \
    --batch_size 32 \
    --visualize

# 大規模評価（数時間）
./run_proteingym_evaluation.sh \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --download-data \
    --batch_size 64 \
    --visualize
```

## 引用

このプロジェクトを使用する場合は、以下を引用してください：

```bibtex
@article{notin2023proteingym,
  title={ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction},
  author={Notin, Pascal and Kollasch, Aaron W and Ritter, Daniel and others},
  journal={bioRxiv},
  year={2023},
  doi={10.1101/2023.12.07.570727}
}
```

## ライセンス

このプロジェクトは元のプロジェクトのライセンスに従います。ProteinGymデータセットの使用については、[ProteinGym公式リポジトリ](https://github.com/OATML-Markslab/ProteinGym)のライセンス条項を確認してください。

## サポート

問題や質問がある場合は、プロジェクトのIssueトラッカーまたは開発チームまでお問い合わせください。
