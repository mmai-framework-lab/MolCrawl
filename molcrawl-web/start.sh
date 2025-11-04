#!/bin/bash

# MolCrawl Web Application - 起動スクリプト
# こちらは古いので折見て削除。

echo "🚀 MolCrawl Dataset Browser を起動しています..."

# 依存関係の確認
if [ ! -d "node_modules" ]; then
  echo "📦 依存関係をインストールしています..."
  npm install
fi

echo "🌐 Webアプリケーションを起動中..."
echo "📊 フロントエンド: http://localhost:3000"
echo "🔌 API サーバー: http://localhost:3001"
echo "📁 ディレクトリ API: http://localhost:3001/api/directory"

# フロントエンドとバックエンドを同時起動
npm run dev
