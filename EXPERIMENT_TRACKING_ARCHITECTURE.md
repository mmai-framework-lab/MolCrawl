# 実験管理システム - アーキテクチャ概要

## システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                       Webブラウザ (UI)                           │
│                    http://localhost:3000                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Dataset      │  │ Experiments  │  │ ZINC Checker │          │
│  │ Browser      │  │ Dashboard    │  │ etc...       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼──────────┐            ┌────────▼─────────┐
│ Express Server   │            │ FastAPI Server   │
│ (molcrawl-web)   │            │ (Experiment API) │
│ Port: 3001       │            │ Port: 8000       │
│                  │            │                  │
│ - File System    │            │ - Experiments    │
│ - ZINC Data      │            │ - Statistics     │
│ - Genome Data    │            │ - Logs           │
└──────────────────┘            └────────┬─────────┘
                                         │
                                         │
                                ┌────────▼─────────┐
                                │                  │
                                │  ExperimentDB    │
                                │  (SQLite)        │
                                │                  │
                                │ experiments.db   │
                                └──────────────────┘
                                         ▲
                                         │
        ┌────────────────────────────────┴───────────────────────────────┐
        │                                                                 │
┌───────┴────────┐    ┌────────────┐    ┌────────────┐    ┌───────────┐
│ Python Scripts │    │ Training   │    │ Evaluation │    │ Data Prep │
│                │    │ Scripts    │    │ Scripts    │    │ Scripts   │
│ - GPT2 Train   │    │ - BERT     │    │ - ProtGym  │    │ - Process │
│ - BERT Train   │    │ - GPN      │    │ - ClinVar  │    │ - Tokenize│
│ - Evaluations  │    │            │    │ - OMIM     │    │           │
└────────────────┘    └────────────┘    └────────────┘    └───────────┘
         │                   │                  │                │
         └───────────────────┴──────────────────┴────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │                  │
                          │ ExperimentTracker│
                          │   (Python SDK)   │
                          │                  │
                          │ - start/complete │
                          │ - log/metrics    │
                          │ - steps tracking │
                          └──────────────────┘
```

## コンポーネント詳細

### 1. フロントエンド層

#### React Web Application (`molcrawl-web/src/`)
- **ExperimentDashboard.js**: 実験管理ダッシュボードコンポーネント
  - 実験一覧の表示
  - フィルタリング機能（ステータス、モデル、データセット）
  - 実験詳細のモーダル表示
  - リアルタイム更新（10秒間隔）
  - 統計情報の表示

- **App.js**: メインアプリケーション
  - タブナビゲーション
  - Experimentsタブの統合

### 2. バックエンド層

#### FastAPI Server (`src/experiment_tracker/api.py`)
- **エンドポイント**:
  - `GET /api/experiments`: 実験一覧取得
  - `GET /api/experiments/{id}`: 実験詳細取得
  - `GET /api/statistics`: 統計情報取得
  - `GET /api/experiments/{id}/logs`: ログ取得
  - `GET /api/experiments/{id}/steps`: ステップ一覧取得

- **機能**:
  - CORS対応（フロントエンドとの通信）
  - クエリパラメータによるフィルタリング
  - 自動ドキュメント生成（Swagger UI）

### 3. データ層

#### SQLite Database (`experiment_data/experiments.db`)

**テーブル構造**:

```sql
-- 実験テーブル
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    model_type TEXT NOT NULL,
    dataset_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    total_duration_seconds REAL,
    config_path TEXT,
    config TEXT,
    results_dir TEXT,
    results TEXT,
    metrics TEXT,
    tags TEXT,
    notes TEXT,
    environment TEXT
);

-- ステップテーブル
CREATE TABLE experiment_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    status TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    duration_seconds REAL,
    command TEXT,
    output_path TEXT,
    error_message TEXT,
    metadata TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- ログテーブル
CREATE TABLE experiment_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    source TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
```

### 4. SDK層

#### ExperimentTracker (`src/experiment_tracker/`)

**主要モジュール**:

- **models.py**: データモデル定義
  - `Experiment`: 実験全体の情報
  - `ExperimentStep`: 実験の各ステップ
  - `ExperimentLog`: ログエントリ
  - 各種Enum（Status, Type, Model, Dataset）

- **database.py**: データベース操作
  - `ExperimentDatabase`: SQLite接続・CRUD操作
  - トランザクション管理
  - クエリビルダー

- **tracker.py**: トラッカー本体
  - `ExperimentTracker`: メインAPI
  - 実験の開始・完了
  - ステップ管理
  - ログ記録

- **helpers.py**: 便利なヘルパー
  - `experiment_context`: コンテキストマネージャー
  - `track_experiment`: デコレータ
  - `simple_track`: シンプルなステップトラッカー

## データフロー

### 1. 実験の記録フロー

```
┌──────────────┐
│ User Script  │
└──────┬───────┘
       │ 1. start_experiment()
       ▼
┌──────────────────┐
│ ExperimentTracker│
└──────┬───────────┘
       │ 2. create Experiment object
       ▼
┌──────────────────┐
│ ExperimentDatabase│
└──────┬───────────┘
       │ 3. INSERT into experiments table
       ▼
┌──────────────────┐
│ SQLite DB        │
│ experiments.db   │
└──────────────────┘
```

### 2. Web UIでの表示フロー

```
┌──────────────────┐
│ Browser          │
│ (React)          │
└──────┬───────────┘
       │ 1. HTTP GET /api/experiments
       ▼
┌──────────────────┐
│ FastAPI Server   │
└──────┬───────────┘
       │ 2. query database
       ▼
┌──────────────────┐
│ ExperimentTracker│
└──────┬───────────┘
       │ 3. SELECT from DB
       ▼
┌──────────────────┐
│ SQLite DB        │
└──────┬───────────┘
       │ 4. return data
       ▼
┌──────────────────┐
│ Browser          │
│ (Display)        │
└──────────────────┘
```

## スケーラビリティ

### 現在の実装（Phase 1）
- ✅ シングルマシン
- ✅ SQLite（軽量、追加インフラ不要）
- ✅ 数千実験まで対応
- ✅ ローカル開発に最適

### 将来の拡張（Phase 2）
- PostgreSQL / MySQLへの移行
- マルチユーザー対応
- 分散実験管理
- クラウドストレージ統合
- WebSocketによるリアルタイム更新

## セキュリティ考慮事項

### 現在
- ローカル開発環境向け
- 認証なし（ローカルホストのみ）
- CORS制限（localhost:3000, localhost:3001のみ）

### 本番環境での推奨事項
- 認証・認可の実装（JWT, OAuth等）
- HTTPSの使用
- データベースのバックアップ
- アクセスログの記録
- 入力バリデーションの強化

## パフォーマンス最適化

### データベース
- インデックスの活用
  - `experiments.status`
  - `experiments.experiment_type`
  - `experiments.created_at`

### API
- ページネーション（limit, offset）
- 遅延読み込み（ログは別エンドポイント）
- キャッシング（将来実装）

### フロントエンド
- 自動更新の制限（10秒間隔）
- 仮想スクロール（大量データ対応）
- 条件付きレンダリング

## 依存関係

### Python
```
fastapi>=0.104.0      # Web framework
uvicorn>=0.24.0       # ASGI server
python-multipart      # Form data handling
sqlite3               # Database (標準ライブラリ)
```

### JavaScript/Node.js
```
react>=19.1.1         # UI framework
react-dom>=19.1.1     # React DOM bindings
express>=4.21.2       # Backend server
cors>=2.8.5           # CORS middleware
```

## ディレクトリ構造

```
riken-dataset-fundational-model/
├── src/
│   └── experiment_tracker/     # 実験管理システムコア
│       ├── __init__.py
│       ├── models.py           # データモデル
│       ├── database.py         # DB操作
│       ├── tracker.py          # トラッカー本体
│       ├── helpers.py          # ヘルパー関数
│       └── api.py              # FastAPI サーバー
├── molcrawl-web/               # Webフロントエンド
│   └── src/
│       ├── ExperimentDashboard.js
│       ├── ExperimentDashboard.css
│       └── App.js              # 統合
├── examples/
│   └── experiment_tracking_example.py
├── experiment_data/            # データ保存先
│   └── experiments.db          # SQLiteデータベース
├── start_api_server.py         # API起動スクリプト
├── start_experiment_system.sh  # 一括起動
├── setup_experiment_system.sh  # セットアップ
├── test_experiment_system.py   # テストスクリプト
├── experiment_requirements.txt # 依存パッケージ
├── EXPERIMENT_TRACKING_README.md
├── EXPERIMENT_TRACKING_QUICKSTART.md
└── EXPERIMENT_TRACKING_ARCHITECTURE.md  # このファイル
```

## 拡張ポイント

### 新しい実験タイプの追加
`src/experiment_tracker/models.py` で Enum に追加:
```python
class ExperimentType(str, Enum):
    # ... 既存
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ABLATION_STUDY = "ablation_study"
```

### 新しいモデルタイプの追加
```python
class ModelType(str, Enum):
    # ... 既存
    TRANSFORMER = "transformer"
    LLAMA = "llama"
```

### カスタムメトリクスの追加
メトリクスは辞書形式で自由に追加可能:
```python
exp.add_metric("custom_score", 0.85)
exp.add_metric("processing_time", 123.45)
```

### 通知機能の追加
```python
# 実験完了時にSlack通知
def on_experiment_complete(experiment_id):
    experiment = tracker.get_experiment(experiment_id)
    send_slack_notification(f"Experiment {experiment.experiment_name} completed!")
```

## まとめ

本システムは、機械学習実験の管理を効率化するために設計された軽量で拡張可能なソリューションです。
既存のワークフローに最小限の変更で統合でき、実験の可視性と再現性を大幅に向上させます。
