# 実験管理システム - クイックスタートガイド

## 📋 必要な手順

### 1. セットアップ（初回のみ）

```bash
# セットアップスクリプトを実行
chmod +x setup_experiment_system.sh
./setup_experiment_system.sh
```

これにより以下が自動的に行われます：

- Python依存パッケージのインストール (FastAPI, Uvicorn)
- Node.js依存パッケージのインストール
- データベースディレクトリの作成
- サンプル実験の実行（オプション）

### 2. システムの起動

```bash
# 一括起動（推奨）
./start_experiment_system.sh
```

または個別に起動:

```bash
# ターミナル1: APIサーバー
python start_api_server.py

# ターミナル2: Webフロントエンド
cd molcrawl-web
npm run dev
```

### 3. アクセス

- **Webインターフェース**: <http://localhost:3000>
- **実験ダッシュボード**: <http://localhost:3000> (Experimentsタブをクリック)
- **API ドキュメント**: <http://localhost:8000/docs>

## 🧪 サンプル実験を実行

```bash
# コンテキストマネージャーの例
python examples/experiment_tracking_example.py --example context

# 手動トラッキングの例
python examples/experiment_tracking_example.py --example manual

# 実験一覧の表示
python examples/experiment_tracking_example.py --example list
```

## 📚 使用方法

### 最小限のコード例

```python
from src.experiment_tracker.helpers import experiment_context
from src.experiment_tracker import ExperimentType, ModelType, DatasetType

with experiment_context(
    name="My First Experiment",
    experiment_type=ExperimentType.TRAINING,
    model_type=ModelType.GPT2,
    dataset_type=DatasetType.PROTEIN_SEQUENCE
) as exp:
    exp.log("INFO", "Starting training")

    exp.start_step("train", "Training model")
    # あなたの訓練コード
    exp.complete_step("train")

    exp.add_metric("accuracy", 0.95)
```

### 既存のスクリプトに統合

既存のPythonスクリプトに数行追加するだけ:

```python
# Before: 通常のスクリプト
def main():
    load_data()
    train_model()
    evaluate()

# After: トラッキング付き
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

        exp.start_step("train", "Train")
        train_model()
        exp.complete_step("train")

        exp.start_step("eval", "Evaluate")
        result = evaluate()
        exp.complete_step("eval")
        exp.add_metric("accuracy", result)
```

## 🧰 テスト

システムが正しく動作するかテスト:

```bash
python test_experiment_system.py
```

## 📖 詳細ドキュメント

詳細な使用方法、API仕様、ベストプラクティスについては:
**[EXPERIMENT_TRACKING_README.md](EXPERIMENT_TRACKING_README.md)** を参照してください。

## 🎯 主な機能

- ✅ 実験の自動記録（開始時刻、終了時刻、実行時間）
- 📊 ステップごとの進捗管理
- 📈 メトリクス（精度、損失など）の記録
- 📝 ログの統合管理
- 🌐 Webブラウザからの閲覧
- 💾 SQLiteベースの軽量データベース
- 🔌 既存コードに簡単統合

## ❓ トラブルシューティング

### ポートが使用中の場合

```bash
# プロセスを確認
lsof -i :8000  # APIサーバー
lsof -i :3000  # Webフロントエンド

# プロセスを停止
kill -9 <PID>
```

### データベースのリセット

```bash
rm -f experiment_data/experiments.db
# 次回起動時に新しいデータベースが作成されます
```

## 📞 サポート

問題が発生した場合は、以下を確認してください:

1. `test_experiment_system.py` でテストを実行
2. `EXPERIMENT_TRACKING_README.md` の詳細ドキュメントを確認
3. ログファイル（コンソール出力）を確認
