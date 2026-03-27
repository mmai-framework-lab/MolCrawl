# MolCrawl Web

MolCrawl Dataset Browser - データセットを探索するためのWebインターフェース

## 必須要件

- Node.js 18以上
- npm
- Python 3.x
- **LEARNING_SOURCE_DIR環境変数の設定（必須）**

## クイックスタート

### 1. 依存関係のインストール

```bash
cd molcrawl-web
npm install
```

### 2. 使用可能なディレクトリを確認

```bash
npm run check-env
```

エラーが表示された場合、使用可能な`learning_source`ディレクトリが表示されます。

### 3. サーバーを起動

**推奨方法（簡単起動スクリプト）**:

```bash
# デフォルトポート（3000, 3001）で起動
LEARNING_SOURCE_DIR="learning_source" ./start-dev.sh

# カスタムポートで起動（フロントエンド=8090, API=8091）
LEARNING_SOURCE_DIR="learning_source" ./start-dev.sh 8090 8091
```

**または環境変数を直接指定**:

```bash
# 環境変数を設定して起動
LEARNING_SOURCE_DIR="learning_source" PORT=8090 API_PORT=8091 npm run dev
```

**手動起動（別々のターミナルで）**:

```bash
# ターミナル1: バックエンドAPI
LEARNING_SOURCE_DIR="learning_source" API_PORT=8091 npm run server

# ターミナル2: フロントエンド（REACT_APP_API_PORTでプロキシ先を指定）
PORT=8090 REACT_APP_API_PORT=8091 npm start
```

**重要**:

- バックエンドAPIサーバーには`API_PORT`を使用
- フロントエンドには`PORT`と`REACT_APP_API_PORT`を使用
- 両方のサーバーが異なるポートで起動していることを確認

デフォルトポート：

- バックエンドAPI: ポート3001
- フロントエンド: ポート3000

起動確認：

```bash
# バックエンドの確認
curl http://localhost:3001/api/health

# ポートの確認
lsof -i :3001  # バックエンド
lsof -i :3000  # フロントエンド
```

#### ポート番号を変更する場合

デフォルトではバックエンドポート3001、フロントエンドポート3000が使用されます。
ポートが既に使用されている場合は、以下の方法でポート番号を変更できます。

**方法1: 環境変数で指定（推奨）**

```bash
# 開発モード: フロントエンドを8090、バックエンドを8091で起動
LEARNING_SOURCE_DIR="learning_source" PORT=8090 API_PORT=8091 npm run dev

# またはターミナルを分けて実行
# ターミナル1: バックエンドを8091で起動
LEARNING_SOURCE_DIR="learning_source" API_PORT=8091 npm run server

# ターミナル2: フロントエンドを8090で起動（自動的に8091のAPIサーバーにプロキシ）
PORT=8090 API_PORT=8091 npm start
```

**方法2: コマンドライン引数で指定（バックエンドのみ）**

```bash
# バックエンドのポート指定
LEARNING_SOURCE_DIR="learning_source" node server.js --port 8091
# または
LEARNING_SOURCE_DIR="learning_source" node server.js -p 8091
```

**重要**:

- `PORT`: React開発サーバー（フロントエンド）のポート番号
- `API_PORT`: Express APIサーバー（バックエンド）のポート番号
- 両方を同じポート番号に設定しないでください（競合します）

**ヘルプを表示**

```bash
node server.js --help
```

### 4. ブラウザでアクセス

- **フロントエンド**: <http://localhost:3000>
- **バックエンドAPI**: <http://localhost:3001/api/health>

## NPMスクリプト

### 開発用

- `npm run dev` - フロントエンドとバックエンドを同時起動（推奨）
- `npm run dev:nfs` - NFSマウント環境用（polling有効）
- `npm start` - フロントエンドのみ起動
- `npm run server` - バックエンドのみ起動
- `npm run check-env` - 環境変数と設定を確認

### ビルド・テスト

- `npm run build` - プロダクションビルド
- `npm test` - テスト実行
- `npm run prod` - ビルド後にサーバー起動
- `npm run prod:serve` - ビルド済みファイルを配信（NFSで安定）
- `npm run serve:build` - ビルドディレクトリを静的ファイルとして配信

### コード品質

- `npm run lint` - ESLintでコードチェック
- `npm run lint:fix` - ESLintで自動修正
- `npm run format` - Prettierでフォーマット
- `npm run format:check` - フォーマットチェックのみ

## NFSマウント環境での起動

NFSマウントされたディレクトリ（例: `/wren`）上でプロジェクトを実行する場合は、特別な設定が必要です。

### 問題と原因

NFSファイルシステムでは `inotify`（ファイル変更監視）が正しく機能しないため、webpack-dev-server が正常に動作しないことがあります。

### 解決方法

#### 方法1: start-dev.sh を使用（推奨・自動検出）

```bash
# NFSマウントを自動検出し、polling モードを有効化
LEARNING_SOURCE_DIR="learning_source" ./start-dev.sh 9090 9091
```

#### 方法2: npm run dev:nfs を使用

```bash
LEARNING_SOURCE_DIR="learning_source" PORT=9090 API_PORT=9091 npm run dev:nfs
```

#### 方法3: 環境変数を明示的に設定

```bash
CHOKIDAR_USEPOLLING=true WATCHPACK_POLLING=true \
LEARNING_SOURCE_DIR="learning_source" PORT=9090 API_PORT=9091 npm run dev
```

#### 方法4: Production Build を使用（最も安定）

```bash
# ビルド
npm run build

# APIサーバー + 静的ファイル配信
LEARNING_SOURCE_DIR="learning_source" API_PORT=9091 npm run prod:serve -- -l 9090
```

詳細は [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) を参照してください。

## 環境変数

### LEARNING_SOURCE_DIR（必須）

データセットのルートディレクトリを指定します。

```bash
export LEARNING_SOURCE_DIR="learning_source"
```

使用可能なディレクトリ:

- `learning_source_202508`
- `learning_source_20251006_genome_all`
- `learning_source_20251020-molecule-nl`

### API_PORT（オプション）

**バックエンドAPIサーバーのポート番号を指定します（デフォルト: 3001）。**

```bash
export API_PORT=8091
```

### PORT（オプション）

**React開発サーバー（フロントエンド）のポート番号を指定します（デフォルト: 3000）。**

```bash
export PORT=8090
```

注:

- バックエンドサーバーは`API_PORT`を優先的に使用します
- コマンドライン引数 `--port` で指定した値が最優先されます
- フロントエンドとバックエンドで異なるポート番号を設定してください

### 永続的に設定する場合

`~/.bashrc`または`~/.zshrc`に追加:

```bash
export LEARNING_SOURCE_DIR="learning_source"
export PORT=8090       # フロントエンド用
export API_PORT=8091   # バックエンドAPI用
```

設定を反映:

```bash
source ~/.bashrc  # または source ~/.zshrc
```

## トラブルシューティング

詳細は[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)を参照してください。

### よくある問題

#### サーバーが起動しない

```bash
❌ ERROR: LEARNING_SOURCE_DIR environment variable is required!
```

**解決方法**: 環境変数を設定してください

```bash
LEARNING_SOURCE_DIR="learning_source" npm run dev
```

#### プロキシエラー: ECONNREFUSED

```
Proxy error: Could not proxy request /api/... from localhost:XXXXX to http://localhost:3001.
(ECONNREFUSED)
```

**原因**: バックエンドサーバー（ポート3001）が起動していない

**解決方法**:

1. バックエンドが起動しているか確認

   ```bash
   lsof -i :3001
   ```

2. 起動していない場合、別ターミナルで起動

   ```bash
   cd molcrawl-web
   LEARNING_SOURCE_DIR="learning_source" npm run server
   ```

3. または、`./start-both.sh`を使用

   ```bash
   export LEARNING_SOURCE_DIR="learning_source"
   ./start-both.sh
   ```

4. `npm run dev`を使う場合、`concurrently`が両方を起動するはずですが、
   エラーが出た場合は手動で起動してください

#### ポートが既に使用されている

```bash
Error: listen EADDRINUSE: address already in use :::3001
```

**解決方法**: 別のポート番号を指定してください

```bash
# フロントエンドとバックエンドを異なるポートで起動
LEARNING_SOURCE_DIR="learning_source" PORT=8090 API_PORT=8091 npm run dev

# または個別に起動
# バックエンドのみ変更
API_PORT=8091 LEARNING_SOURCE_DIR="learning_source" npm run server

# フロントエンドのみ変更
PORT=8090 npm start
```

**重要**: `PORT`と`API_PORT`に同じ値を設定しないでください。React開発サーバーとAPIサーバーが競合します。

#### 500エラーが発生する

バックエンドサーバーが起動していない可能性があります。

**解決方法**: `npm start`ではなく`npm run dev`または`./start-both.sh`を使用してください

## 機能

### 📊 データセット準備進捗モニタリング

5つのデータセット準備スクリプトの進捗状況をリアルタイムで監視できます：

- **Protein Sequence (Uniprot)** - 3ステップ
  - Uniprot Download → FASTA to Raw → Tokenization
- **Genome Sequence (RefSeq)** - 4ステップ
  - RefSeq Download → FASTA to Raw → Tokenizer Training → Raw to Parquet
- **RNA (CellxGene)** - 5ステップ
  - Build List → Download → H5AD to Loom → Tokenization → Vocabulary
- **Molecule NL (SMolInstruct)** - 2ステップ
  - Dataset Download/Copy → Tokenization & Processing
- **Compounds (OrganiX13)** - 3ステップ
  - OrganiX13 Download → SMILES & Scaffolds Tokenization → Statistics

各データセットの進捗状況は、マーカーファイルと出力ファイルの存在で自動判定されます。

#### 🚀 準備スクリプト実行機能（新機能）

各データセットの「準備進捗」カードから、準備スクリプトを直接実行できます：

- **Phase 01ボタン**: データダウンロードと基本前処理スクリプトを実行
  - 例: `01-protein_sequence-prepare.sh`
- **Phase 02ボタン**: GPT-2用データセット準備スクリプトを実行
  - 例: `02-protein_sequence-prepare-gpt2.sh`

**機能詳細**:

- ✅ ワンクリックでスクリプトを実行開始
- 📋 リアルタイムでログをモーダル表示（2秒間隔で自動更新）
- ⏹️ 実行中のスクリプトを停止可能
- 📊 実行状態（PID、実行時間、ステータス）を表示
- 🔄 スクリプト完了後に自動で進捗を再取得

**使用方法**:

1. 各データセットタブの「準備進捗」カードを確認
2. 「▶ Phase 01」または「▶ Phase 02」ボタンをクリック
3. ログモーダルが開き、リアルタイムでログを表示
4. スクリプトが完了するまで待機、または「⏹ 停止」ボタンで停止
5. モーダルを閉じても、スクリプトはバックグラウンドで実行継続

#### 使用方法（進捗確認）

1. Webブラウザで <http://localhost:3000> にアクセス
2. 「Preparation」タブをクリック
3. 各データセットの進捗を確認
4. 自動更新オプションで5秒ごとにリフレッシュ可能

## プロジェクト構造

```
molcrawl-web/
├── api/                    # バックエンドAPI
│   ├── directory.js       # ディレクトリAPI
│   ├── dataset-progress.js # データセット準備進捗API
│   ├── genome-species.js  # ゲノム種API
│   └── zinc-checker.js    # ZINC20データチェッカー
├── src/                    # Reactフロントエンド
│   ├── App.js             # メインアプリケーション
│   ├── DatasetProgress.js # データセット準備進捗コンポーネント
│   ├── ExperimentDashboard.js
│   ├── GenomeSpeciesList.js
│   └── ZincChecker.js
├── public/                 # 静的ファイル
├── server.js              # Expressサーバー
├── package.json           # 依存関係
└── check-config.js        # 設定チェックスクリプト
```

## API エンドポイント

### ヘルスチェック

- `GET /api/health` - サーバーステータス

### ディレクトリ操作

- `GET /api/directory` - ルートディレクトリ構造取得
- `GET /api/directory/expand?path=<path>` - ディレクトリ展開
- `GET /api/directory/tree?maxDepth=5` - 完全ツリー取得

### ゲノムデータ

- `GET /api/genome/species` - ゲノム種リスト取得
- `GET /api/genome/species/category?category=<category>` - カテゴリ別種リスト

### ZINC20データ

- `GET /api/zinc/check` - ZINC20データチェック
- `GET /api/zinc/count` - ZINC20データ件数取得

### データセット準備進捗

- `GET /api/dataset-progress` - 全データセットの準備進捗取得
- `GET /api/dataset-progress/:datasetKey` - 特定データセットの詳細進捗取得
  - `datasetKey`: `protein_sequence`, `genome_sequence`, `rna`, `molecule_nat_lang`, `compounds`

### 準備スクリプト実行（新機能）

- `GET /api/preparation-runner/scripts` - 利用可能なスクリプト一覧
- `POST /api/preparation-runner/start` - 準備スクリプトを実行
  - Body: `{ dataset: 'protein_sequence', phase: 'phase01' }`
- `GET /api/preparation-runner/status/:dataset/:phase` - 実行状態を取得
- `GET /api/preparation-runner/log/:dataset/:phase?lines=200` - 実行ログを取得
- `POST /api/preparation-runner/stop` - スクリプトを停止
  - Body: `{ dataset: 'protein_sequence', phase: 'phase01' }`
- `GET /api/preparation-runner/all-status` - すべての実行状態を取得

## 開発

### ESLint設定

`.eslintrc.json`に設定があります。詳細は[ESLINT_SETUP.md](./ESLINT_SETUP.md)を参照。

### Prettier設定

`.prettierrc.json`に設定があります。
