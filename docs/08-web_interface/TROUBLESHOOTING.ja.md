# MolCrawl Web - トラブルシューティングガイド

## 必須: 環境変数の設定

**重要**: このアプリケーションは`LEARNING_SOURCE_DIR`環境変数が必須です。設定しないと起動しません。

## 起動方法

### ステップ1: 使用可能なディレクトリを確認

```bash
cd molcrawl-web
npm run check-env
```

これにより、使用可能な`learning_source`ディレクトリが表示されます。

### ステップ2: 環境変数を設定して起動

#### 方法1: 1行で起動（推奨）

```bash
cd molcrawl-web
LEARNING_SOURCE_DIR="learning_source" npm run dev
```

#### 方法2: 環境変数をエクスポートして起動

```bash
cd molcrawl-web
export LEARNING_SOURCE_DIR="learning_source"
npm run dev
```

#### 方法3: .bashrcまたは.zshrcに追加（永続化）

```bash
# ~/.bashrc または ~/.zshrc に追加
export LEARNING_SOURCE_DIR="learning_source"

# 設定を反映
source ~/.bashrc  # または source ~/.zshrc
```

---

## NFSマウント環境での起動

### 問題の概要

NFSマウントされたディレクトリ（例: `/wren`）上でプロジェクトを実行する場合、webpack-dev-serverが正常に動作しないことがあります。

**症状:**

- `npm run dev` を実行すると "Compiled successfully!" と表示されるが、ポートがリッスンされない
- `ss -tlnp | grep 9090` でポートが表示されない
- `curl http://localhost:9090` で "接続を拒否されました" エラー

**原因:**

- NFSファイルシステムでは `inotify`（ファイル変更監視機能）が正しく機能しない
- webpack-dev-serverは内部でinotifyを使用してファイル変更を監視している
- inotifyが機能しないと、HTTPサーバーが正しく起動しない場合がある

### 解決方法

#### 方法1: start-dev.shスクリプトを使用（推奨）

`start-dev.sh` スクリプトはNFSマウントを自動検出し、必要な設定を有効にします。

```bash
cd molcrawl-web
LEARNING_SOURCE_DIR="learning_source" ./start-dev.sh 9090 9091
```

#### 方法2: 環境変数を手動で設定

polling モードを有効にして起動します：

```bash
cd molcrawl-web
CHOKIDAR_USEPOLLING=true WATCHPACK_POLLING=true \
LEARNING_SOURCE_DIR="learning_source" \
PORT=9090 API_PORT=9091 npm run dev
```

#### 方法3: npm run dev:nfs を使用

package.json に NFS 対応スクリプトが用意されています：

```bash
cd molcrawl-web
LEARNING_SOURCE_DIR="learning_source" \
PORT=9090 API_PORT=9091 npm run dev:nfs
```

#### 方法4: Production Build を使用

開発サーバーの代わりにビルド済みファイルを配信します（ホットリロード無し）：

```bash
cd molcrawl-web

# ビルド（初回または変更後に実行）
npm run build

# APIサーバーとフロントエンドを同時起動
LEARNING_SOURCE_DIR="learning_source" \
API_PORT=9091 npm run prod:serve -- -l 9090
```

または別々に起動：

```bash
# ターミナル1: APIサーバー
LEARNING_SOURCE_DIR="learning_source" API_PORT=9091 npm run server

# ターミナル2: フロントエンド（静的ファイル配信）
npx serve build -l 9090
```

### NFS環境の設定ファイル

`.env.development` に以下の設定が含まれています：

```bash
# NFS Mount Environment Settings
CHOKIDAR_USEPOLLING=true
WATCHPACK_POLLING=true
```

これらの設定は自動的に読み込まれますが、環境によっては明示的に環境変数として渡す必要がある場合があります。

### NFS環境の確認方法

現在のディレクトリがNFSマウントかどうかを確認：

```bash
df -T .
```

出力に `nfs` または `nfs4` が含まれていればNFSマウントです。

---

## よくあるエラーと解決方法

### エラー: `LEARNING_SOURCE_DIR environment variable is required!`

環境変数が設定されていません。

**解決方法**:

```bash
# 使用可能なディレクトリを確認
npm run check-env

# 環境変数を設定して起動
LEARNING_SOURCE_DIR="learning_source" npm run dev
```

### エラー: `Specified LEARNING_SOURCE_DIR does not exist!`

指定したディレクトリが存在しません。

**解決方法**:

```bash
# 正しいディレクトリ名を確認
ls -d ../learning_source*

# 正しい名前で再起動
LEARNING_SOURCE_DIR="正しいディレクトリ名" npm run dev
```

### エラー: `ECONNREFUSED localhost:3001`

バックエンドサーバーが起動していません。

**解決方法**:

- `npm start`ではなく`npm run dev`を使用してください
- または別ターミナルで`npm run server`を実行してください

### ポートが既に使用されている

```bash
# ポート3001を使用しているプロセスを確認
lsof -i :3001

# プロセスを終了
kill -9 <PID>

# または一括でポートを解放
fuser -k 3000/tcp 3001/tcp
```

### 開発サーバーが起動するがポートがリッスンされない

NFSマウント環境の問題です。上記「NFSマウント環境での起動」セクションを参照してください。

---

## アクセスURL

起動後、以下のURLにアクセスできます：

- **フロントエンド**: <http://localhost:3000>
- **バックエンドAPI**: <http://localhost:3001/api/health>
- **ディレクトリAPI**: <http://localhost:3001/api/directory>
