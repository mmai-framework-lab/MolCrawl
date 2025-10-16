# 実験管理システム (Experiment Tracking System)

## 概要

本システムは、GPT-2とBERTモデルの各工程（データ準備、訓練、評価、可視化）を一元管理し、実験の進捗、結果、ログを統合的に記録・閲覧するためのシステムです。

## 主な機能

- ✅ **実験の自動記録**: 各実験の開始・終了時刻、実行時間、ステータスを自動記録
- 📊 **ステップ単位の管理**: 実験内の各ステップ（データロード、前処理、訓練など）を個別に追跡
- 📈 **メトリクスの記録**: 精度、損失などのメトリクスを自動保存
- 📝 **ログの統合管理**: すべてのログを一箇所に集約
- 🌐 **Web UI**: ブラウザから実験一覧、詳細、統計情報を閲覧可能
- 💾 **軽量データベース**: SQLiteベースで追加のインフラ不要
- 🔌 **簡単統合**: 既存のスクリプトに数行追加するだけで利用可能

## システム構成

```
実験管理システム
├── src/experiment_tracker/     # コアモジュール
│   ├── models.py              # データモデル定義
│   ├── database.py            # SQLiteデータベース
│   ├── tracker.py             # トラッカー本体
│   ├── helpers.py             # ヘルパー関数
│   └── api.py                 # FastAPI バックエンド
├── molcrawl-web/              # Webフロントエンド
│   └── src/ExperimentDashboard.js
├── experiment_data/           # データベース保存先
│   └── experiments.db         # SQLiteデータベース
└── examples/                  # サンプルコード
    └── experiment_tracking_example.py
```

## クイックスタート

### 1. 依存パッケージのインストール

```bash
# FastAPIとUvicornをインストール
pip install fastapi uvicorn sqlalchemy

# Webフロントエンドの依存関係（初回のみ）
cd molcrawl-web
npm install
cd ..
```

### 2. システムの起動

#### 方法A: 自動起動（推奨）

```bash
# APIサーバーとWebフロントエンドを同時起動
chmod +x start_experiment_system.sh
./start_experiment_system.sh
```

#### 方法B: 個別起動

```bash
# ターミナル1: APIサーバー起動
python start_api_server.py

# ターミナル2: Webフロントエンド起動
cd molcrawl-web
npm run dev
```

### 3. アクセス

- **Web UI**: http://localhost:3000
- **実験ダッシュボード**: http://localhost:3000 (Experimentsタブ)
- **API ドキュメント**: http://localhost:8000/docs

## 使用方法

### パターン1: コンテキストマネージャー（推奨）

最も柔軟で詳細な管理が可能です。

```python
from src.experiment_tracker.helpers import experiment_context
from src.experiment_tracker import ExperimentType, ModelType, DatasetType

with experiment_context(
    name="GPT2 ProteinGym Training",
    experiment_type=ExperimentType.TRAINING,
    model_type=ModelType.GPT2,
    dataset_type=DatasetType.PROTEINGYM,
    config={"epochs": 10, "batch_size": 32},
    config_path="configs/gpt2_train.yaml"
) as exp:
    
    # ログを記録
    exp.log("INFO", "Training started")
    
    # ステップ1: データロード
    exp.start_step("data_loading", "Load dataset")
    train_data = load_dataset("proteingym_train.csv")
    exp.complete_step("data_loading", output_path="data/processed/train.pt")
    
    # ステップ2: 訓練
    exp.start_step("training", "Train model")
    model = train_model(train_data)
    exp.complete_step("training", output_path="models/gpt2_proteingym.pt")
    
    # ステップ3: 評価
    exp.start_step("evaluation", "Evaluate model")
    accuracy = evaluate_model(model)
    exp.complete_step("evaluation")
    
    # 結果とメトリクスを記録
    exp.add_result("model_path", "models/gpt2_proteingym.pt")
    exp.add_metric("accuracy", accuracy)
    exp.add_metric("loss", 0.123)
    
    exp.log("INFO", f"Training completed. Accuracy: {accuracy:.4f}")
```

### パターン2: デコレータ

関数全体を実験として記録します。

```python
from src.experiment_tracker.helpers import track_experiment
from src.experiment_tracker import ExperimentType, ModelType, DatasetType

@track_experiment(
    name="BERT ClinVar Evaluation",
    experiment_type=ExperimentType.EVALUATION,
    model_type=ModelType.BERT,
    dataset_type=DatasetType.CLINVAR
)
def run_evaluation(config):
    # 評価処理
    model = load_model(config['model_path'])
    results = evaluate(model, config['test_data'])
    
    # 返り値の数値はメトリクスとして自動記録
    return {
        "accuracy": results['accuracy'],
        "f1_score": results['f1'],
        "model_path": config['model_path']  # 数値以外は結果として記録
    }

# 実行
results = run_evaluation({"model_path": "models/bert.pt", "test_data": "data/test.csv"})
```

### パターン3: 手動トラッキング

最も低レベルなAPIで、細かい制御が可能です。

```python
from src.experiment_tracker import ExperimentTracker, ExperimentType, ModelType, DatasetType

tracker = ExperimentTracker()

# 実験開始
exp_id = tracker.start_experiment(
    name="GPT2 RNA Data Preparation",
    experiment_type=ExperimentType.DATA_PREPARATION,
    model_type=ModelType.GPT2,
    dataset_type=DatasetType.RNA,
    tags=["preprocessing", "tokenization"]
)

try:
    # ステップ実行
    tracker.start_step(exp_id, "download", "Download raw data")
    download_data()
    tracker.complete_step(exp_id, "download", output_path="data/raw/rna.csv")
    
    tracker.start_step(exp_id, "preprocess", "Preprocess and tokenize")
    preprocess_data()
    tracker.complete_step(exp_id, "preprocess", output_path="data/processed/rna.pt")
    
    # 実験完了
    tracker.complete_experiment(
        exp_id,
        results={"output_dir": "data/processed/"},
        metrics={"num_samples": 100000, "vocab_size": 512}
    )
    
except Exception as e:
    tracker.fail_experiment(exp_id, str(e))
    raise
```

## 既存スクリプトへの統合例

### シェルスクリプトからの利用

```bash
#!/bin/bash

# Pythonラッパーを作成
cat > /tmp/track_experiment.py << 'EOF'
from src.experiment_tracker import ExperimentTracker, ExperimentType, ModelType, DatasetType
import sys
import subprocess

tracker = ExperimentTracker()
exp_id = tracker.start_experiment(
    name=sys.argv[1],
    experiment_type=ExperimentType(sys.argv[2]),
    model_type=ModelType(sys.argv[3]),
    dataset_type=DatasetType(sys.argv[4])
)

tracker.start_step(exp_id, "execution", "Execute script")
try:
    result = subprocess.run(sys.argv[5:], check=True, capture_output=True, text=True)
    tracker.complete_step(exp_id, "execution")
    tracker.complete_experiment(exp_id)
    print(result.stdout)
except Exception as e:
    tracker.fail_step(exp_id, "execution", str(e))
    tracker.fail_experiment(exp_id, str(e))
    sys.exit(1)
EOF

# 使用例
python /tmp/track_experiment.py \
    "GPT2 Training" \
    "training" \
    "gpt2" \
    "protein_sequence" \
    python bert/main.py --config configs/bert_train.yaml
```

### Pythonスクリプトの最小限の変更

```python
# Before
def main():
    load_data()
    train_model()
    evaluate_model()

if __name__ == "__main__":
    main()
```

```python
# After
from src.experiment_tracker.helpers import experiment_context
from src.experiment_tracker import ExperimentType, ModelType, DatasetType

def main():
    with experiment_context(
        name="My Experiment",
        experiment_type=ExperimentType.TRAINING,
        model_type=ModelType.GPT2,
        dataset_type=DatasetType.PROTEIN_SEQUENCE
    ) as exp:
        exp.start_step("data", "Load data")
        load_data()
        exp.complete_step("data")
        
        exp.start_step("train", "Train model")
        train_model()
        exp.complete_step("train")
        
        exp.start_step("eval", "Evaluate")
        evaluate_model()
        exp.complete_step("eval")

if __name__ == "__main__":
    main()
```

## API エンドポイント

### 実験一覧の取得

```bash
curl "http://localhost:8000/api/experiments?status=completed&model_type=gpt2"
```

### 実験詳細の取得

```bash
curl "http://localhost:8000/api/experiments/{experiment_id}"
```

### 統計情報の取得

```bash
curl "http://localhost:8000/api/statistics"
```

### ログの取得

```bash
curl "http://localhost:8000/api/experiments/{experiment_id}/logs?level=ERROR"
```

## データモデル

### ExperimentStatus
- `pending`: 未実行
- `running`: 実行中
- `completed`: 完了
- `failed`: 失敗
- `cancelled`: キャンセル
- `skipped`: スキップ

### ExperimentType
- `data_preparation`: データ準備
- `training`: 訓練
- `evaluation`: 評価
- `visualization`: 可視化
- `inference`: 推論

### ModelType
- `gpt2`: GPT-2モデル
- `bert`: BERTモデル
- `gpn`: GPNモデル

### DatasetType
- `protein_sequence`: タンパク質配列
- `genome_sequence`: ゲノム配列
- `compounds`: 化合物
- `rna`: RNA
- `molecule_related_natural_language`: 分子自然言語
- `proteingym`: ProteinGym
- `clinvar`: ClinVar
- `omim`: OMIM
- `cosmic`: COSMIC

## サンプルの実行

複数のサンプル実験を実行してシステムをテストできます:

```bash
# 各種パターンのサンプルを実行
python examples/experiment_tracking_example.py --example context
python examples/experiment_tracking_example.py --example manual
python examples/experiment_tracking_example.py --example failure

# 実験一覧を表示
python examples/experiment_tracking_example.py --example list
```

## トラブルシューティング

### データベースのリセット

```bash
rm -f experiment_data/experiments.db
# システムを再起動すると新しいデータベースが作成されます
```

### APIサーバーのログ確認

```bash
# start_api_server.pyを直接実行して詳細ログを確認
python start_api_server.py
```

### データベースの直接確認

```bash
sqlite3 experiment_data/experiments.db
> .tables
> SELECT * FROM experiments ORDER BY created_at DESC LIMIT 5;
> .quit
```

## ベストプラクティス

1. **実験名を明確に**: `"GPT2 ProteinGym Large 20251006"` のように、モデル、データセット、日付を含める
2. **タグを活用**: `tags=["benchmark", "ablation", "production"]` で実験を分類
3. **メトリクスは数値のみ**: `metrics` には float や int のみを記録
4. **ログレベルを適切に**: INFO, WARNING, ERROR, DEBUG を使い分ける
5. **定期的にエクスポート**: 重要な実験は JSON でバックアップ

```python
tracker.export_experiment_json(exp_id, f"backups/experiment_{exp_id}.json")
```

## 今後の拡張予定

- [ ] 実験の比較機能
- [ ] グラフ・チャートの自動生成
- [ ] 実験の複製・再実行機能
- [ ] Slackなどへの通知連携
- [ ] TensorBoardとの統合
- [ ] 複数ユーザーサポート
- [ ] クラウドストレージへのバックアップ

## ライセンス

本システムは riken-dataset-fundational-model プロジェクトの一部です。
