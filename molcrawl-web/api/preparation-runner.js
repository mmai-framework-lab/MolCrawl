/**
 * Preparation Script Runner API
 * データ準備スクリプト（01_*, 02_*）を実行し、ログをリアルタイム表示するAPI
 */

const express = require('express');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const router = express.Router();

// プロジェクトルートディレクトリ（molcrawl-webの親ディレクトリ = riken-dataset-fundational-model）
const PROJECT_ROOT = path.resolve(__dirname, '../..');

// 起動時にパスを確認
console.log('[Preparation Runner] Initialized');
console.log('[Preparation Runner] PROJECT_ROOT:', PROJECT_ROOT);
console.log('[Preparation Runner] Workflows directory:', path.join(PROJECT_ROOT, 'workflows'));
console.log('[Preparation Runner] Workflows exists:', fs.existsSync(path.join(PROJECT_ROOT, 'workflows')));

// 実行中のプロセスを管理
const runningProcesses = new Map();

// データセットごとの準備スクリプトマッピング
const PREPARATION_SCRIPTS = {
  protein_sequence: {
    phase01: {
      script: '01-protein_sequence-prepare.sh',
      logPattern: 'protein-sequence-preparation-*.log',
      logDir: 'protein_sequence/logs',
    },
    phase02: {
      script: '02-protein_sequence-prepare-gpt2.sh',
      logPattern: 'protein_sequence-prepare-gpt2-*.log',
      logDir: 'protein_sequence/logs',
    },
  },
  genome_sequence: {
    phase01: {
      script: '01-genome_sequence-prepare.sh',
      logPattern: 'genome_sequence-preparation-*.log',
      logDir: 'genome_sequence/logs',
    },
    phase02: {
      script: '02-genome_sequence-prepare-gpt2.sh',
      logPattern: 'genome_sequence-prepare-gpt2-*.log',
      logDir: 'genome_sequence/logs',
    },
  },
  compounds: {
    phase01: {
      script: '01-compounds_prepare.sh',
      logPattern: 'compounds-preparation-*.log',
      logDir: 'compounds/logs',
    },
    phase02: {
      script: '02-compounds-prepare-gpt2.sh',
      logPattern: 'compounds-prepare-gpt2-*.log',
      logDir: 'compounds/logs',
    },
  },
  compounds_guacamol: {
    phase01: {
      script: '01-compounds_guacamol-prepare.sh',
      logPattern: 'compounds-guacamol-preparation-*.log',
      logDir: 'compounds/logs',
    },
    phase02: {
      script: '02-compounds-prepare-gpt2.sh',
      logPattern: 'compounds-prepare-gpt2-*.log',
      logDir: 'compounds/logs',
    },
  },
  molecule_nat_lang: {
    phase01: {
      script: '01-molecule_nat_lang-prepare.sh',
      logPattern: 'molecule_nat_lang-preparation-*.log',
      logDir: 'molecule_nat_lang/logs',
    },
    phase02: {
      script: '02-molecule_nat_lang-prepare-gpt2.sh',
      logPattern: 'molecule_nat_lang-prepare-gpt2-*.log',
      logDir: 'molecule_nat_lang/logs',
    },
  },
  rna: {
    phase01: {
      script: '01-rna-prepare.sh',
      logPattern: 'rna-preparation-*.log',
      logDir: 'rna/logs',
    },
    phase02: {
      script: '02-rna-prepare-gpt2.sh',
      logPattern: 'rna-prepare-gpt2-*.log',
      logDir: 'rna/logs',
    },
  },
};

/**
 * 最新のログファイルを検出
 * スクリプトが生成したログファイルを見つける
 */
function findLatestLogFile(learningSourceDir, logDir, logPattern, sinceTime = null) {
  const glob = require('glob');
  const logDirPath = path.join(PROJECT_ROOT, learningSourceDir, logDir);

  if (!fs.existsSync(logDirPath)) {
    return null;
  }

  // globパターンでログファイルを検索
  const pattern = path.join(logDirPath, logPattern);
  const files = glob.sync(pattern);

  if (files.length === 0) {
    return null;
  }

  // 指定時刻以降に作成されたファイルをフィルタ（sinceTimeがある場合のみ）
  let targetFiles = files;
  if (sinceTime) {
    targetFiles = files.filter(file => {
      const stat = fs.statSync(file);
      return stat.mtimeMs >= sinceTime;
    });

    if (targetFiles.length === 0) {
      return null;
    }
  }

  // 最新のファイルを返す
  targetFiles.sort((a, b) => {
    return fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs;
  });

  return targetFiles[0];
}

/**
 * スクリプトが実際に実行中かチェック
 * プロセス名でgrep検索して確認
 */
function checkScriptRunning(scriptName) {
  const { execSync } = require('child_process');
  try {
    // pgrep -f でスクリプト名を含むプロセスを検索
    const result = execSync(`pgrep -f "${scriptName}"`, { encoding: 'utf8' });
    const pids = result.trim().split('\n').filter(p => p);
    return pids.length > 0 ? pids : null;
  } catch (error) {
    // pgrepが見つからない場合はfalseを返す
    return null;
  }
}

/**
 * GET /api/preparation-runner/scripts
 * 利用可能な準備スクリプト一覧を取得
 */
router.get('/scripts', (req, res) => {
  try {
    const scripts = Object.entries(PREPARATION_SCRIPTS).map(([dataset, phases]) => ({
      dataset,
      phases: {
        phase01: phases.phase01,
        phase02: phases.phase02,
      },
    }));

    res.json({
      success: true,
      scripts,
    });
  } catch (error) {
    console.error('Error getting scripts list:', error);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * POST /api/preparation-runner/start
 * 準備スクリプトを実行
 * Body: { dataset: 'protein_sequence', phase: 'phase01' }
 */
router.post('/start', (req, res) => {
  try {
    const { dataset, phase } = req.body;

    console.log(`[Preparation Runner] Start request - dataset: ${dataset}, phase: ${phase}`);

    if (!dataset || !phase) {
      return res.status(400).json({
        success: false,
        error: 'dataset and phase are required',
      });
    }

    const scripts = PREPARATION_SCRIPTS[dataset];
    if (!scripts || !scripts[phase]) {
      console.error(`[Preparation Runner] Script configuration not found - dataset: ${dataset}, phase: ${phase}`);
      return res.status(404).json({
        success: false,
        error: `Script not found for dataset: ${dataset}, phase: ${phase}`,
        availableDatasets: Object.keys(PREPARATION_SCRIPTS),
      });
    }

    const scriptConfig = scripts[phase];
    const scriptName = scriptConfig.script;
    const scriptPath = path.join(PROJECT_ROOT, 'workflows', scriptName);

    console.log(`[Preparation Runner] PROJECT_ROOT: ${PROJECT_ROOT}`);
    console.log(`[Preparation Runner] Script name: ${scriptName}`);
    console.log(`[Preparation Runner] Script path: ${scriptPath}`);

    if (!fs.existsSync(scriptPath)) {
      console.error(`[Preparation Runner] Script file not found: ${scriptPath}`);
      return res.status(404).json({
        success: false,
        error: `Script file not found: ${scriptPath}`,
        scriptName,
        projectRoot: PROJECT_ROOT,
      });
    }

    // 既に実行中かチェック
    const processKey = `${dataset}_${phase}`;
    const learningSourceDir = process.env.LEARNING_SOURCE_DIR || 'learning_source_20260106';

    // 実際にスクリプトが実行中かチェック
    const runningPids = checkScriptRunning(scriptName);
    if (runningPids) {
      console.log(`[Preparation Runner] Script already running with PIDs: ${runningPids.join(', ')}`);

      // ログファイルを検出
      const existingLogFile = findLatestLogFile(
        learningSourceDir,
        scriptConfig.logDir,
        scriptConfig.logPattern
      );

      // プロセス情報を更新または作成
      if (!runningProcesses.has(processKey)) {
        runningProcesses.set(processKey, {
          pid: parseInt(runningPids[0]),
          dataset,
          phase,
          scriptName,
          scriptLogFile: existingLogFile,
          logDetectionInfo: {
            logDir: scriptConfig.logDir,
            logPattern: scriptConfig.logPattern,
            startTime: null,
          },
          learningSourceDir,
          startTime: existingLogFile ? new Date(fs.statSync(existingLogFile).mtimeMs) : new Date(),
          status: 'running',
        });
      }

      return res.json({
        success: true,
        alreadyRunning: true,
        message: 'スクリプトは既に実行中です。既存のログを表示します。',
        processKey,
        pid: runningPids[0],
        logFile: existingLogFile,
        hasLog: !!existingLogFile,
      });
    }

    // Mapにはあるが実プロセスがない場合は削除
    if (runningProcesses.has(processKey)) {
      console.log(`[Preparation Runner] Removing stale process entry: ${processKey}`);
      runningProcesses.delete(processKey);
    }

    // 新規プロセスを起動
    const scriptStartTime = Date.now(); // ログファイル検出用のタイムスタンプ

    console.log(`[Preparation Runner] Starting script: ${scriptPath}`);
    console.log(`[Preparation Runner] CWD: ${PROJECT_ROOT}`);
    console.log(`[Preparation Runner] LEARNING_SOURCE_DIR: ${learningSourceDir}`);

    const child = spawn('bash', [scriptPath], {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        LEARNING_SOURCE_DIR: learningSourceDir,
      },
      detached: false,
    });

    if (!child.pid) {
      throw new Error('Failed to spawn process');
    }

    console.log(`[Preparation Runner] Process started with PID: ${child.pid}`);

    // スクリプトが作成するログファイルを検出するための情報を保存
    const logDetectionInfo = {
      logDir: scriptConfig.logDir,
      logPattern: scriptConfig.logPattern,
      startTime: scriptStartTime,
    };

    // プロセス情報を保存
    runningProcesses.set(processKey, {
      pid: child.pid,
      dataset,
      phase,
      scriptName,
      scriptLogFile: null, // 後で検出したログファイルパスを設定
      logDetectionInfo,
      learningSourceDir,
      startTime: new Date(),
      status: 'running',
    });

    // ログファイルを検出するタイマーを開始（1秒後に開始、その後3秒ごと）
    setTimeout(() => {
      detectAndUpdateLogFile(processKey, learningSourceDir, logDetectionInfo);
      const detectInterval = setInterval(() => {
        const processInfo = runningProcesses.get(processKey);
        if (!processInfo || processInfo.status !== 'running' || processInfo.scriptLogFile) {
          clearInterval(detectInterval);
          return;
        }
        detectAndUpdateLogFile(processKey, learningSourceDir, logDetectionInfo);
      }, 3000);
    }, 1000);

    // 標準出力をコンソールに出力（デバッグ用）
    child.stdout.on('data', (data) => {
      console.log(`[Script ${processKey}]`, data.toString());
    });

    // 標準エラーをコンソールに出力（デバッグ用）
    child.stderr.on('data', (data) => {
      console.error(`[Script ${processKey}]`, data.toString());
    });

    // プロセス終了時の処理
    child.on('close', (code) => {
      const processInfo = runningProcesses.get(processKey);
      if (processInfo) {
        processInfo.status = code === 0 ? 'completed' : 'failed';
        processInfo.exitCode = code;
        processInfo.endTime = new Date();

        // 最後にログファイルを検出
        if (!processInfo.scriptLogFile) {
          detectAndUpdateLogFile(processKey, learningSourceDir, logDetectionInfo);
        }
      }
      console.log(`[Preparation Runner] Process ${processKey} exited with code ${code}`);
    });

    child.on('error', (error) => {
      console.error(`[Preparation Runner] Process ${processKey} error:`, error);
      const processInfo = runningProcesses.get(processKey);
      if (processInfo) {
        processInfo.status = 'error';
        processInfo.error = error.message;
      }
    });

    console.log(`[Preparation Runner] Successfully started ${scriptName}`);

    res.json({
      success: true,
      processKey,
      pid: child.pid,
      scriptPath,
      message: `Started ${scriptName}. Log file will be detected automatically.`,
      logDetection: {
        logDir: scriptConfig.logDir,
        logPattern: scriptConfig.logPattern,
      },
    });
  } catch (error) {
    console.error('[Preparation Runner] Error in start endpoint:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      stack: error.stack,
    });
  }
});

/**
 * ログファイルを検出してプロセス情報を更新
 */
function detectAndUpdateLogFile(processKey, learningSourceDir, logDetectionInfo) {
  try {
    const logFile = findLatestLogFile(
      learningSourceDir,
      logDetectionInfo.logDir,
      logDetectionInfo.logPattern,
      logDetectionInfo.startTime
    );

    if (logFile) {
      const processInfo = runningProcesses.get(processKey);
      if (processInfo && !processInfo.scriptLogFile) {
        processInfo.scriptLogFile = logFile;
        console.log(`[Preparation Runner] Detected log file for ${processKey}: ${logFile}`);
      }
    }
  } catch (error) {
    console.error(`[Preparation Runner] Error detecting log file for ${processKey}:`, error);
  }
}

/**
 * GET /api/preparation-runner/status/:dataset/:phase
 * 準備スクリプトの実行状態を取得
 */
router.get('/status/:dataset/:phase', (req, res) => {
  try {
    const { dataset, phase } = req.params;
    const processKey = `${dataset}_${phase}`;

    const processInfo = runningProcesses.get(processKey);
    if (!processInfo) {
      return res.json({
        success: true,
        running: false,
        processKey,
      });
    }

    res.json({
      success: true,
      running: processInfo.status === 'running',
      processKey,
      ...processInfo,
      duration: processInfo.endTime
        ? processInfo.endTime - processInfo.startTime
        : Date.now() - processInfo.startTime,
    });
  } catch (error) {
    console.error('Error getting process status:', error);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * GET /api/preparation-runner/log/:dataset/:phase
 * 準備スクリプトのログを取得（最新N行）
 */
router.get('/log/:dataset/:phase', (req, res) => {
  try {
    const { dataset, phase } = req.params;
    const { lines = 100 } = req.query;
    const processKey = `${dataset}_${phase}`;

    const processInfo = runningProcesses.get(processKey);

    if (!processInfo) {
      return res.status(404).json({
        success: false,
        error: 'Process not found',
        processKey,
      });
    }

    // スクリプトが作成したログファイルを優先
    let logFilePath = processInfo.scriptLogFile;

    // ログファイルがまだ検出されていない場合、再度検出を試みる
    if (!logFilePath && processInfo.logDetectionInfo) {
      logFilePath = findLatestLogFile(
        processInfo.learningSourceDir,
        processInfo.logDetectionInfo.logDir,
        processInfo.logDetectionInfo.logPattern,
        processInfo.logDetectionInfo.startTime
      );

      if (logFilePath) {
        processInfo.scriptLogFile = logFilePath;
        console.log(`[Preparation Runner] Late detection of log file for ${processKey}: ${logFilePath}`);
      }
    }

    if (!logFilePath) {
      return res.status(404).json({
        success: false,
        error: 'Log file not yet created or not found',
        processKey,
        hint: 'The script may still be initializing. Please wait a moment and try again.',
      });
    }

    if (!fs.existsSync(logFilePath)) {
      return res.status(404).json({
        success: false,
        error: 'Log file does not exist',
        path: logFilePath,
      });
    }

    // ログファイルの末尾N行を取得
    const logContent = fs.readFileSync(logFilePath, 'utf8');
    const logLines = logContent.split('\n');
    const tailLines = logLines.slice(-parseInt(lines));

    res.json({
      success: true,
      processKey,
      logFilePath,
      lines: tailLines.length,
      content: tailLines.join('\n'),
      status: processInfo.status,
      fileSize: fs.statSync(logFilePath).size,
    });
  } catch (error) {
    console.error('Error getting log:', error);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * POST /api/preparation-runner/stop
 * 準備スクリプトを停止
 * Body: { dataset: 'protein_sequence', phase: 'phase01' }
 */
router.post('/stop', (req, res) => {
  try {
    const { dataset, phase } = req.body;
    const processKey = `${dataset}_${phase}`;

    const processInfo = runningProcesses.get(processKey);
    if (!processInfo) {
      return res.status(404).json({
        success: false,
        error: 'Process not found',
      });
    }

    if (processInfo.status !== 'running') {
      return res.status(400).json({
        success: false,
        error: `Process is not running (status: ${processInfo.status})`,
      });
    }

    // プロセスを停止
    process.kill(processInfo.pid, 'SIGTERM');

    processInfo.status = 'stopped';
    processInfo.endTime = new Date();

    res.json({
      success: true,
      processKey,
      message: 'Process stopped',
    });
  } catch (error) {
    console.error('Error stopping process:', error);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * GET /api/preparation-runner/all-status
 * すべての準備スクリプトの実行状態を取得
 */
router.get('/all-status', (req, res) => {
  try {
    const allStatus = {};

    for (const [processKey, processInfo] of runningProcesses.entries()) {
      allStatus[processKey] = {
        ...processInfo,
        duration: processInfo.endTime
          ? processInfo.endTime - processInfo.startTime
          : Date.now() - processInfo.startTime,
      };
    }

    res.json({
      success: true,
      processes: allStatus,
      count: runningProcesses.size,
    });
  } catch (error) {
    console.error('Error getting all status:', error);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

module.exports = router;
