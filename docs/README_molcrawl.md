# MolCrawl Dataset Browser 🧬

**Fundamental Models Dataset Explorer** - RIKEN MolCrawl プロジェクトのデータセット構造をブラウザで視覚化

## 概要

このWebアプリケーションは、`learning_source_202508`ディレクトリの構造を視覚化し、データセットの内容を効率的に探索するためのツールです。

### 主な機能

- 📁 **ディレクトリツリー表示**: 階層構造を直感的に表示
- 🔍 **折りたたみ可能**: フォルダを展開/折りたたみ
- 📊 **ファイル情報**: ファイルサイズとディレクトリ内アイテム数
- 🔄 **リアルタイム更新**: 最新のディレクトリ構造を取得
- 📱 **レスポンシブデザイン**: モバイル対応

## 技術スタック

### フロントエンド
- **React.js** 19.1.1 - モダンなユーザーインターフェース
- **CSS3** - カスタムスタイリング
- **Fetch API** - バックエンドとの通信

### バックエンド
- **Node.js + Express** - RESTful API サーバー
- **CORS** - クロスオリジンリクエスト対応
- **File System API** - ディレクトリスキャン

## インストール

```bash
# プロジェクトディレクトリに移動
cd /data2/user/MolCrawl/riken-dataset-fundational-model/molcrawl-web

# 依存関係をインストール
npm install
```

## 使用方法

### 1. 開発モード（推奨）

```bash
# フロントエンドとバックエンドを同時起動
npm run dev
```

または起動スクリプトを使用:

```bash
./start.sh
```

### 2. 個別起動

```bash
# バックエンドのみ（ポート3001）
npm run server

# フロントエンドのみ（ポート3000）
npm start
```

### 3. プロダクションビルド

```bash
# ビルドして本番サーバー起動
npm run prod
```

## アクセス

- **Webインターフェース**: http://localhost:3000
- **API サーバー**: http://localhost:3001
- **ヘルスチェック**: http://localhost:3001/api/health

## API エンドポイント

### ディレクトリ構造取得
```
GET /api/directory?path={directory_path}
```

### 子ディレクトリ展開
```
GET /api/directory/expand?path={directory_path}
```

### ヘルスチェック
```
GET /api/health
```

## 対象ディレクトリ

`/data2/user/MolCrawl/riken-dataset-fundational-model/learning_source_202508/`

### 含まれるデータセット
- **cellxgene/**: 単細胞RNAシーケンスデータ
- **refseq/**: ゲノム配列データ（RefSeq）
- **uniprot/**: タンパク質配列データ（UniProt）

## セキュリティ

- パス検証により、指定ディレクトリ外へのアクセスを制限
- CORS設定でローカルホストからのアクセスのみ許可
- ファイル読み取り専用（書き込み操作なし）

---

### クイックスタート

```bash
cd molcrawl-web
npm install
./start.sh
# ブラウザで http://localhost:3000 を開く
```
