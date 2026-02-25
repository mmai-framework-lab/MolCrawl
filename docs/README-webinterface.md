# 🧬 RIKEN Dataset Foundational Models - Web Interface

HuggingFace事前学習データセットの情報を可視化・管理するためのWebインターフェースです。

## 📊 概要

learning_source genome_sequence, molecule_nl, protein_sequence, rna）のtraining_ready_hf_datasetを包括的に表示・管理できます。

### 🎯 主な機能

- **データセット概要**: 全モデルのデータセット統計情報
- **モデル別詳細**: 各モデルのHuggingFaceデータセット詳細
- **ファイル構造表示**: ディレクトリとファイルの階層表示
- **容量・統計情報**: データサイズ、ファイル数、更新日時
- **リアルタイム更新**: API経由でのリアルタイムデータ取得

## 🚀 クイックスタート

### 必要環境

- Node.js (v14以上)
- npm
- 適切なファイルアクセス権限

### 1. 起動

```bash
# 実行権限を付与
chmod +x start-new.sh

# サービス起動
./start-new.sh
```

### 2. アクセス

- **フロントエンド**: <http://localhost:3000>
- **API**: <http://localhost:3001>

## 📁 ディレクトリ構造

```text
molcrawl-web/
├── api/                          # バックエンドAPI
│   ├── server.js                 # メインサーバー
│   ├── routes.js                 # APIルート定義
│   ├── dataset-info.js           # データセット情報API
│   ├── directory.js              # ディレクトリ構造API
│   └── package.json
├── frontend/                     # Reactフロントエンド
│   ├── src/
│   │   ├── components/
│   │   │   ├── DatasetInfo.js    # HFデータセット表示
│   │   │   ├── DatasetInfo.css
│   │   │   ├── DirectoryViewer.js
│   │   │   └── DirectoryViewer.css
│   │   ├── App.js               # メインアプリ
│   │   ├── App.css
│   │   └── index.js
│   ├── public/
│   └── package.json
└── start-new.sh                 # 起動スクリプト
```

## 🛠️ API エンドポイント

### HuggingFaceデータセット情報

| エンドポイント                    | 説明                       |
| --------------------------------- | -------------------------- |
| `GET /api/datasets/all`           | 全モデルのデータセット情報 |
| `GET /api/datasets/:modelName`    | 特定モデルの詳細情報       |
| `GET /api/datasets/summary/stats` | 統計サマリー               |

### ディレクトリ構造

| エンドポイント              | 説明                 |
| --------------------------- | -------------------- |
| `GET /api/directory`        | ディレクトリ構造取得 |
| `GET /api/directory/expand` | ディレクトリ展開     |

## 🎨 画面構成

### 1. HFデータセットタブ 🤗

#### 概要画面

- **統計カード**: 総モデル数、データありモデル数、総ファイル数、総容量
- **モデルグリッド**: 各モデルの詳細情報カード
  - compounds (🧪): 化合物分子構造
  - genome_sequence (🧬): ゲノムDNA配列
  - molecule_nl (📝): 分子の自然言語記述
  - protein_sequence (🧬): タンパク質アミノ酸配列
  - rna (🧬): RNA核酸配列

#### モデル詳細画面

- **基本情報**: データセット数、ファイル数、容量、更新日時
- **データセット一覧**: 各データセットの詳細情報
- **サンプルファイル**: 各データセット内のファイル例

### 2. ディレクトリブラウザタブ 📁

- **階層表示**: フォルダ・ファイルの階層構造
- **展開機能**: クリックでフォルダ内容を表示
- **サイズ表示**: ファイル・フォルダの容量表示

## 🔧 カスタマイズ

### データソースの変更

`api/dataset-info.js`の`LEARNING_SOURCE_BASE`を変更:

```javascript
const LEARNING_SOURCE_BASE = "/path/to/your/learning_source";
```

### モデル設定の追加

`MODELS`オブジェクトに新しいモデルを追加:

```javascript
const MODELS = {
  // 既存のモデル...
  new_model: {
    name: "New Model",
    description: "New model description",
    icon: "🔬",
  },
};
```

## 📋 トラブルシューティング

### よくある問題

1. **ポート使用エラー**

   ```bash
   # ポート確認
   lsof -i :3000
   lsof -i :3001

   # プロセス終了
   kill -9 <PID>
   ```

2. **権限エラー**

   ```bash
   # 実行権限付与
   chmod +x start-new.sh

   # ディレクトリアクセス権確認
   ls -la /wren/matsubara/riken-dataset-fundational-model/learning_source
   ```

3. **依存関係エラー**

   ```bash
   # キャッシュクリア後再インストール
   cd api && rm -rf node_modules package-lock.json && npm install
   cd ../frontend && rm -rf node_modules package-lock.json && npm install
   ```

### ログ確認

```bash
# APIログ
tail -f logs/api.log

# フロントエンドログ
tail -f logs/frontend.log
```

## 🔄 開発・デバッグ

### 開発モード起動

```bash
# API開発モード
cd api && npm run dev

# フロントエンド開発モード（別ターミナル）
cd frontend && npm start
```

### デバッグ情報

ブラウザの開発者ツールでネットワークタブを確認し、API呼び出しの状況を監視できます。

## 📈 パフォーマンス最適化

- **大容量データセット**: ページネーション実装推奨
- **リアルタイム更新**: WebSocket実装でリアルタイム監視可能
- **キャッシュ**: Redis等でAPI結果のキャッシュ可能

## 🤝 貢献

1. フォーク
2. フィーチャーブランチ作成
3. 変更をコミット
4. プルリクエスト作成

## 📄 ライセンス

MIT License

---

**🧬 RIKEN Dataset Foundational Models Management Interface**
_Developed for efficient dataset management and visualization_
