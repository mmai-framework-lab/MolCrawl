const express = require('express');
const cors = require('cors');
const path = require('path');
const { getDirectoryStructure, expandDirectory, getFullDirectoryTree, checkZincData, getZincDataCounts } = require('./api/directory');

const app = express();
const PORT = process.env.PORT || 3001;

// model_dirの値をサーバー起動時に確認
console.log('Server starting with model_dir configuration...');
console.log('Current working directory:', process.cwd());

// CORS設定
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:3000'],
  methods: ['GET', 'POST'],
  credentials: true
}));

// JSON解析
app.use(express.json());

// 静的ファイル配信（現在のディレクトリから）
app.use(express.static(__dirname));

// API Routes
app.get('/api/directory', getDirectoryStructure);
app.get('/api/directory/expand', expandDirectory);
app.get('/api/directory/tree', getFullDirectoryTree);
app.get('/api/zinc/check', checkZincData);
app.get('/api/zinc/count', getZincDataCounts);

// ヘルスチェック
app.get('/api/health', (req, res) => {
  res.json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    message: 'MolCrawl Web API Server is running',
    endpoints: [
      '/api/directory - ルートディレクトリ構造取得',
      '/api/directory/expand?path=<path>&recursive=true - ディレクトリ展開',
      '/api/directory/tree?maxDepth=5&includeFiles=true - 完全ツリー取得',
      '/api/zinc/check - ZINC20データチェック',
      '/api/zinc/count - ZINC20データ件数取得'
    ]
  });
});

// エラーハンドリング
app.use((error, req, res, next) => {
  console.error('Server Error:', error);
  res.status(500).json({
    error: 'Internal Server Error',
    message: error.message,
    timestamp: new Date().toISOString()
  });
});

// 404ハンドリング
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.method} ${req.url} not found`,
    timestamp: new Date().toISOString()
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`📁 Directory API: http://localhost:${PORT}/api/directory`);
  console.log(`🌳 Full Tree API: http://localhost:${PORT}/api/directory/tree`);
  console.log(`🌐 Test page: http://localhost:${PORT}/test.html`);
});

module.exports = app;
