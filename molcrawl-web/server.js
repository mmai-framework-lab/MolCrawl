const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

// 環境変数チェック（API読み込み前に実行）
if (!process.env.LEARNING_SOURCE_DIR) {
  console.error('');
  console.error('❌ ERROR: LEARNING_SOURCE_DIR environment variable is required!');
  console.error('');
  console.error('Please set it before starting the server:');
  console.error('  export LEARNING_SOURCE_DIR="learning_source_202508"');
  console.error('  npm run dev');
  console.error('');
  console.error('Or run with inline environment variable:');
  console.error('  LEARNING_SOURCE_DIR="learning_source_202508" npm run dev');
  console.error('');
  process.exit(1);
}

// LEARNING_SOURCE_DIRディレクトリの存在チェック
const learningSourcePath = path.join(__dirname, '..', process.env.LEARNING_SOURCE_DIR);
if (!fs.existsSync(learningSourcePath)) {
  console.error('');
  console.error('❌ ERROR: LEARNING_SOURCE_DIR directory does not exist!');
  console.error('');
  console.error(`Specified directory: ${process.env.LEARNING_SOURCE_DIR}`);
  console.error(`Expected path: ${learningSourcePath}`);
  console.error('');
  console.error('Available directories in project root:');
  try {
    const projectRoot = path.join(__dirname, '..');
    const directories = fs.readdirSync(projectRoot, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory() && dirent.name.startsWith('learning_'))
      .map(dirent => `  - ${dirent.name}`);

    if (directories.length > 0) {
      console.error(directories.join('\n'));
      console.error('');
      console.error('Please use one of the above directories or check the directory name spelling.');
    } else {
      console.error('  No learning_* directories found');
    }
  } catch (err) {
    console.error('  Unable to list directories:', err.message);
  }
  console.error('');
  process.exit(1);
}

const { getDirectoryStructure, expandDirectory, getFullDirectoryTree, checkZincData, getZincDataCounts, validateDirectoryExists } = require('./api/directory');
const { getGenomeSpeciesList, getGenomeSpeciesByCategory } = require('./api/genome-species');
const datasetProgressRouter = require('./api/dataset-progress');
const gpt2TrainingStatusRouter = require('./api/gpt2-training-status');
const gpt2InferenceRouter = require('./api/gpt2-inference');
const bertTrainingStatusRouter = require('./api/bert-training-status');
const trainingProcessStatusRouter = require('./api/training-process-status');
const { getLogsList, getAllLogsOverview, getLogContent, getTailLog } = require('./api/logs');
const { getGpuInfo, getGpuXmlInfo } = require('./api/gpu-resources');

const app = express();
const PORT = process.env.PORT || 3001;

// model_dirの値をサーバー起動時に確認
console.log('✅ Server starting with configuration:');
console.log('   LEARNING_SOURCE_DIR:', process.env.LEARNING_SOURCE_DIR);
console.log('   Working directory:', process.cwd());

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

// API Routes - ディレクトリに依存するエンドポイントにはバリデーションを適用
app.get('/api/directory', validateDirectoryExists, getDirectoryStructure);
app.get('/api/directory/expand', validateDirectoryExists, expandDirectory);
app.get('/api/directory/tree', validateDirectoryExists, getFullDirectoryTree);

// 画像API
app.use('/api/images', require('./api/images'));
app.get('/api/zinc/check', validateDirectoryExists, checkZincData);
app.get('/api/zinc/count', validateDirectoryExists, getZincDataCounts);
app.get('/api/genome/species', validateDirectoryExists, getGenomeSpeciesList);
app.get('/api/genome/species/category', validateDirectoryExists, getGenomeSpeciesByCategory);
app.use('/api/dataset-progress', validateDirectoryExists, datasetProgressRouter);
app.use('/api/gpt2-training-status', validateDirectoryExists, gpt2TrainingStatusRouter);
app.use('/api/gpt2-inference', validateDirectoryExists, gpt2InferenceRouter);
app.use('/api/bert-training-status', validateDirectoryExists, bertTrainingStatusRouter);
app.use('/api/training-process-status', trainingProcessStatusRouter);
app.use('/api/preparation-runner', require('./api/preparation-runner'));

// ログAPI
app.get('/api/logs/list', validateDirectoryExists, getLogsList);
app.get('/api/logs/overview', validateDirectoryExists, getAllLogsOverview);
app.get('/api/logs/content', validateDirectoryExists, getLogContent);
app.get('/api/logs/tail', validateDirectoryExists, getTailLog);

// GPU リソースAPI
app.get('/api/gpu/info', getGpuInfo);
app.get('/api/gpu/xml', getGpuXmlInfo);

// ヘルスチェック
app.get('/api/health', (req, res) => {
  const fsSync = require('fs');
  const learningSourcePath = path.join(__dirname, '..', process.env.LEARNING_SOURCE_DIR);
  const directoryExists = fsSync.existsSync(learningSourcePath);

  res.json({
    status: directoryExists ? 'OK' : 'ERROR',
    timestamp: new Date().toISOString(),
    message: directoryExists
      ? 'MolCrawl Web API Server is running'
      : `LEARNING_SOURCE_DIR directory '${process.env.LEARNING_SOURCE_DIR}' not found`,
    configuration: {
      learning_source_dir: process.env.LEARNING_SOURCE_DIR,
      directory_path: learningSourcePath,
      directory_exists: directoryExists
    },
    endpoints: [
      '/api/directory - ルートディレクトリ構造取得',
      '/api/directory/expand?path=<path>&recursive=true - ディレクトリ展開',
      '/api/directory/tree?maxDepth=5&includeFiles=true - 完全ツリー取得',
      '/api/zinc/check - ZINC20データチェック',
      '/api/zinc/count - ZINC20データ件数取得',
      '/api/genome/species - ゲノム種リスト取得',
      '/api/genome/species/category?category=<category> - カテゴリ別種リスト取得',
      '/api/dataset-progress - 全データセット準備進捗取得',
      '/api/dataset-progress/:datasetKey - 特定データセット詳細進捗取得',
      '/api/gpt2-training-status - 全GPT-2モデルの学習状況取得',
      '/api/gpt2-training-status/:dataset - 特定データセットのGPT-2学習状況',
      '/api/gpt2-training-status/:dataset/:size - 特定モデルの詳細情報',
      '/api/bert-training-status - 全BERTモデルの学習状況取得',
      '/api/bert-training-status/:dataset - 特定データセットのBERT学習状況',
      '/api/bert-training-status/:dataset/:size - 特定BERTモデルの詳細情報',
      '/api/training-process-status - 学習プロセス稼働状況チェック',
      '/api/logs/list?modelPath=<path> - 指定モデルのログファイル一覧取得',
      '/api/logs/overview - 全モデルのログファイル概要取得',
      '/api/logs/content?logPath=<path> - ログファイル内容取得',
      '/api/logs/tail?logPath=<path>&lines=100 - ログファイルの末尾取得',
      '/api/images/:modelType - 指定モデルの画像一覧取得',
      '/api/images/serve/:modelType/:filename - 画像ファイル配信',
      '/api/images/thumbnail/:modelType/:filename - サムネイル画像配信',
      '/api/gpu/info - GPUリソース情報取得（nvidia-smi）',
      '/api/gpu/xml - GPUリソース詳細情報取得（nvidia-smi XML）'
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
