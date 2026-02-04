const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

// コマンドライン引数の解析
function parseArgs() {
  const args = process.argv.slice(2);
  const parsed = {};
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--port' || args[i] === '-p') {
      parsed.port = parseInt(args[i + 1], 10);
      i++;
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.log('');
      console.log('MolCrawl Web Server');
      console.log('');
      console.log('Usage:');
      console.log('  node server.js [options]');
      console.log('');
      console.log('Options:');
      console.log('  -p, --port <port>    Specify the port number (default: 3001)');
      console.log('  -h, --help           Show this help message');
      console.log('');
      console.log('Environment Variables:');
      console.log('  API_PORT                 API server port number (recommended, can be overridden by --port)');
      console.log('  PORT                     Fallback port number if API_PORT is not set');
      console.log('  LEARNING_SOURCE_DIR      Required: Learning source directory name');
      console.log('');
      console.log('Examples:');
      console.log('  API_PORT=3002 node server.js');
      console.log('  node server.js --port 3002');
      console.log('  LEARNING_SOURCE_DIR="learning_source_202508" API_PORT=8091 npm run dev');
      console.log('  LEARNING_SOURCE_DIR="learning_source_202508" PORT=8090 API_PORT=8091 npm run dev');
      console.log('');
      process.exit(0);
    }
  }
  
  return parsed;
}

const cmdArgs = parseArgs();

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
const bertInferenceRouter = require('./api/bert-inference');
const trainingProcessStatusRouter = require('./api/training-process-status');
const { getLogsList, getAllLogsOverview, getLogContent, getTailLog } = require('./api/logs');
const { getGpuInfo, getGpuXmlInfo } = require('./api/gpu-resources');

const app = express();
// ポート番号の優先順位: コマンドライン引数 > API_PORT環境変数 > デフォルト(3001)
// PORTはReact開発サーバー専用なので、APIサーバーでは使用しない
const PORT = cmdArgs.port || process.env.API_PORT || 3001;

// ポート番号のバリデーション
if (isNaN(PORT) || PORT < 1 || PORT > 65535) {
  console.error('');
  console.error('❌ ERROR: Invalid port number!');
  console.error('');
  console.error(`Specified port: ${PORT}`);
  console.error('Port number must be between 1 and 65535.');
  console.error('');
  console.error('Use --help for usage information.');
  console.error('');
  process.exit(1);
}

// model_dirの値をサーバー起動時に確認
console.log('✅ Server starting with configuration:');
console.log('   LEARNING_SOURCE_DIR:', process.env.LEARNING_SOURCE_DIR);
console.log('   PORT:', PORT);
console.log('   Working directory:', process.cwd());

// CORS設定
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:3000', `http://localhost:${PORT}`, `http://127.0.0.1:${PORT}`],
  methods: ['GET', 'POST'],
  credentials: true
}));

// JSON解析
app.use(express.json());

// 静的ファイル配信（現在のディレクトリから）
app.use(express.static(__dirname));

// Production: buildディレクトリの静的ファイル配信
const buildPath = path.join(__dirname, 'build');
if (fs.existsSync(buildPath)) {
  console.log('📦 Serving static files from build directory');
  app.use(express.static(buildPath));
}

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
app.use('/api/bert-inference', validateDirectoryExists, bertInferenceRouter);
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
      '/api/gpt2-inference - GPT-2推論実行',
      '/api/gpt2-inference/config/:dataset - GPT-2推論設定取得',
      '/api/bert-training-status - 全BERTモデルの学習状況取得',
      '/api/bert-training-status/:dataset - 特定データセットのBERT学習状況',
      '/api/bert-training-status/:dataset/:size - 特定BERTモデルの詳細情報',
      '/api/bert-inference - BERT穴埋め推論実行',
      '/api/bert-inference/config/:dataset - BERT推論設定取得',
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

// Production: React SPA のフォールバック（APIルート以外）
// すべての非APIリクエストをindex.htmlに転送
const buildPath2 = path.join(__dirname, 'build');
if (fs.existsSync(buildPath2)) {
  app.get('*', (req, res) => {
    // APIリクエストは404を返す
    if (req.path.startsWith('/api/')) {
      return res.status(404).json({
        error: 'Not Found',
        message: `API Route ${req.method} ${req.url} not found`,
        timestamp: new Date().toISOString()
      });
    }
    // それ以外はindex.htmlを返す（React Router対応）
    res.sendFile(path.join(buildPath2, 'index.html'));
  });
} else {
  // 404ハンドリング（開発モード用）
  app.use((req, res) => {
    res.status(404).json({
      error: 'Not Found',
      message: `Route ${req.method} ${req.url} not found`,
      timestamp: new Date().toISOString()
    });
  });
}

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`📁 Directory API: http://localhost:${PORT}/api/directory`);
  console.log(`🌳 Full Tree API: http://localhost:${PORT}/api/directory/tree`);
  console.log(`🌐 Test page: http://localhost:${PORT}/test.html`);
});

module.exports = app;
