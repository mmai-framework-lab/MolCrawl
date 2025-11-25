/**
 * Dataset Preparation Progress API
 * 各データセット準備スクリプトの進捗状況を監視するAPI
 */

const express = require('express');
const fs = require('fs');
const path = require('path');

const router = express.Router();

/**
 * 各データセットの定義
 * マーカーファイルと出力ファイル/ディレクトリで進捗を判定
 */
const DATASETS = {
  protein_sequence: {
    name: 'Protein Sequence (Uniprot)',
    baseDir: 'protein_sequence',
    steps: [
      {
        id: 'download',
        name: 'Uniprot Download',
        marker: 'download_complete.marker',
        checkFiles: [],
        outputDirs: ['downloads', 'uniprot_data'],
      },
      {
        id: 'fasta_to_raw',
        name: 'FASTA to Raw Conversion',
        marker: 'fasta_to_raw_complete.marker',
        checkFiles: ['raw_files'],
        outputDirs: ['raw_files'],
      },
      {
        id: 'tokenize',
        name: 'Tokenization to Parquet',
        marker: 'tokenize_to_parquet_complete.marker',
        checkFiles: ['parquet_files/train.parquet'],
        outputDirs: ['parquet_files'],
        outputFiles: ['parquet_files/train.parquet', 'parquet_files/valid.parquet'],
      },
    ],
    outputs: {
      plot: '../assets/img/protein_sequence_tokenized_lengths_dist.png',
      statistics: null,
      parquetFiles: 'parquet_files',
    },
  },
  genome_sequence: {
    name: 'Genome Sequence (RefSeq)',
    baseDir: 'genome_sequence',
    steps: [
      {
        id: 'download',
        name: 'RefSeq Download',
        marker: 'download_complete.marker',
        checkFiles: ['download_dir'],
        outputDirs: ['download_dir'],
      },
      {
        id: 'fasta_to_raw',
        name: 'FASTA to Raw Conversion',
        marker: 'fasta_to_raw_complete.marker',
        checkFiles: ['raw_files'],
        outputDirs: ['raw_files'],
      },
      {
        id: 'train_tokenizer',
        name: 'Tokenizer Training',
        marker: 'train_tokenizer_complete.marker',
        checkFiles: ['spm_tokenizer.model'],
        outputDirs: ['.'],
        outputFiles: ['spm_tokenizer.model', 'spm_tokenizer.vocab'],
      },
      {
        id: 'raw_to_parquet',
        name: 'Raw to Parquet Conversion',
        marker: 'raw_to_parquet_complete.marker',
        checkFiles: ['parquet_files'],
        outputDirs: ['parquet_files'],
        outputFiles: ['parquet_files/train.parquet', 'parquet_files/valid.parquet'],
      },
    ],
    outputs: {
      plot: '../assets/img/genome_sequence_tokenized_lengths_dist.png',
      statistics: null,
      parquetFiles: 'parquet_files',
    },
  },
  rna: {
    name: 'RNA (CellxGene)',
    baseDir: 'rna',
    steps: [
      {
        id: 'build_list',
        name: 'Build Dataset List',
        marker: 'build_list_complete.marker',
        checkFiles: [],
        outputDirs: ['.'],
        outputFiles: ['dataset_list.json', 'dataset_metadata.csv'],
      },
      {
        id: 'download',
        name: 'Dataset Download',
        marker: 'download_complete.marker',
        checkFiles: [],
        outputDirs: ['downloads', 'h5ad_files'],
      },
      {
        id: 'h5ad_to_loom',
        name: 'H5AD to Loom Conversion',
        marker: 'h5ad_to_loom_complete.marker',
        checkFiles: [],
        outputDirs: ['loom_files'],
      },
      {
        id: 'tokenize',
        name: 'Tokenization',
        marker: 'tokenize_complete.marker',
        checkFiles: ['parquet_files'],
        outputDirs: ['parquet_files'],
        outputFiles: ['parquet_files/train.parquet', 'parquet_files/valid.parquet'],
      },
      {
        id: 'vocab',
        name: 'Vocabulary Generation',
        marker: null,
        checkFiles: ['gene_vocab.json'],
        outputDirs: ['.'],
        outputFiles: ['gene_vocab.json', 'gene_list_with_stats.tsv'],
      },
    ],
    outputs: {
      plot: '../assets/img/rna_tokenized_lengths_dist.png',
      statistics: 'rna_stats.json',
      geneList: 'gene_list_with_stats.tsv',
      parquetFiles: 'parquet_files',
    },
  },
  molecule_nl: {
    name: 'Molecule Natural Language (SMolInstruct)',
    baseDir: 'molecule_nl',
    steps: [
      {
        id: 'download',
        name: 'Dataset Download/Copy',
        marker: null,
        checkFiles: ['osunlp/SMolInstruct/dataset_info.json'],
        outputDirs: ['osunlp/SMolInstruct'],
      },
      {
        id: 'tokenize',
        name: 'Tokenization & Processing',
        marker: null,
        checkFiles: ['molecule_related_natural_language_tokenized.parquet', 'parquet_files'],
        outputDirs: ['.', 'parquet_files'],
        outputFiles: ['molecule_related_natural_language_tokenized.parquet', 'parquet_files/train.parquet', 'parquet_files/valid.parquet'],
      },
    ],
    outputs: {
      plot: '../assets/img/molecule_nl_tokenized_train_lengths_dist.png',
      statistics: null,
      parquetFiles: 'parquet_files',
    },
  },
  compounds: {
    name: 'Compounds (OrganiX13)',
    baseDir: 'compounds',
    steps: [
      {
        id: 'download',
        name: 'ZINC20 Download',
        marker: 'organix13/download_complete.marker',
        checkFiles: [],
        outputDirs: ['zinc20'],
      },
      {
        id: 'tokenize',
        name: 'SMILES & Scaffolds Tokenization',
        marker: 'organix13/tokenized_complete.marker',
        checkFiles: ['organix13/OrganiX13_tokenized.parquet'],
        outputDirs: ['organix13'],
        outputFiles: ['organix13/OrganiX13_tokenized.parquet'],
      },
      {
        id: 'statistics',
        name: 'Statistics Generation',
        marker: 'organix13/stats_complete.marker',
        checkFiles: ['parquet_files'],
        outputDirs: ['organix13', 'parquet_files'],
        outputFiles: ['organix13/statistics.json', 'parquet_files/train.parquet', 'parquet_files/valid.parquet'],
      },
    ],
    outputs: {
      plot: '../assets/img/compounds_tokenized_SMILES_lengths_dist.png',
      scaffoldPlot: '../assets/img/compounds_tokenized_Scaffolds_lengths_dist.png',
      statistics: null,
      parquetFiles: 'parquet_files',
    },
  },
};

/**
 * ファイル/ディレクトリの存在確認
 */
function checkExists(fullPath) {
  try {
    return fs.existsSync(fullPath);
  } catch (error) {
    return false;
  }
}

/**
 * ディレクトリ内にファイルが存在するか確認
 */
function checkDirHasFiles(dirPath) {
  try {
    if (!fs.existsSync(dirPath)) return false;
    const stats = fs.statSync(dirPath);
    if (!stats.isDirectory()) return false;
    const files = fs.readdirSync(dirPath);
    return files.length > 0;
  } catch (error) {
    return false;
  }
}

/**
 * 単一ステップの状態をチェック
 */
function checkStepStatus(learningSourcePath, dataset, step) {
  const baseDir = path.join(learningSourcePath, dataset.baseDir);

  // マーカーファイルがあればそれを優先
  if (step.marker) {
    const markerPath = path.join(baseDir, step.marker);
    if (checkExists(markerPath)) {
      return 'completed';
    }
  }

  // checkFilesで詳細確認
  if (step.checkFiles && step.checkFiles.length > 0) {
    let allExist = true;
    for (const file of step.checkFiles) {
      const filePath = path.join(baseDir, file);
      const exists = checkExists(filePath);

      if (!exists) {
        allExist = false;
        break;
      }

      // ディレクトリの場合、中身があるか確認
      try {
        const stats = fs.statSync(filePath);
        if (stats.isDirectory() && !checkDirHasFiles(filePath)) {
          allExist = false;
          break;
        }
      } catch (error) {
        allExist = false;
        break;
      }
    }

    if (allExist) {
      return 'completed';
    }
  }

  return 'pending';
}

/**
 * データセットの全体進捗を取得
 */
function getDatasetProgress(learningSourcePath, datasetKey, dataset) {
  const steps = dataset.steps.map((step) => {
    const status = checkStepStatus(learningSourcePath, dataset, step);
    return {
      id: step.id,
      name: step.name,
      status: status,
    };
  });

  const completedSteps = steps.filter((s) => s.status === 'completed').length;
  const totalSteps = steps.length;
  const progressPercent = Math.round((completedSteps / totalSteps) * 100);

  // 出力ファイルの存在確認
  const outputs = {};
  const projectRoot = path.dirname(path.dirname(learningSourcePath));

  if (dataset.outputs.plot) {
    const plotPath = path.join(projectRoot, dataset.outputs.plot);
    outputs.plot = checkExists(plotPath);
  }

  if (dataset.outputs.scaffoldPlot) {
    const scaffoldPlotPath = path.join(projectRoot, dataset.outputs.scaffoldPlot);
    outputs.scaffoldPlot = checkExists(scaffoldPlotPath);
  }

  if (dataset.outputs.statistics) {
    const statsPath = path.join(
      learningSourcePath,
      dataset.baseDir,
      dataset.outputs.statistics
    );
    outputs.statistics = checkExists(statsPath);
  }

  if (dataset.outputs.geneList) {
    const geneListPath = path.join(
      learningSourcePath,
      dataset.baseDir,
      dataset.outputs.geneList
    );
    outputs.geneList = checkExists(geneListPath);
  }

  return {
    name: dataset.name,
    baseDir: dataset.baseDir,
    steps: steps,
    progress: {
      completed: completedSteps,
      total: totalSteps,
      percent: progressPercent,
    },
    outputs: outputs,
    status:
      completedSteps === totalSteps
        ? 'completed'
        : completedSteps > 0
          ? 'in_progress'
          : 'not_started',
  };
}

/**
 * GET /api/dataset-progress
 * 全データセットの進捗状況を取得
 */
router.get('/', (req, res) => {
  const learningSourceDir = process.env.LEARNING_SOURCE_DIR;

  if (!learningSourceDir) {
    return res.status(500).json({
      error: 'LEARNING_SOURCE_DIR environment variable is not set',
    });
  }

  // プロジェクトルートからの絶対パス構築
  const projectRoot = path.resolve(__dirname, '..', '..');
  const learningSourcePath = path.join(projectRoot, learningSourceDir);

  if (!checkExists(learningSourcePath)) {
    return res.status(404).json({
      error: 'Learning source directory not found',
      path: learningSourcePath,
    });
  }

  const progress = {};

  for (const [key, dataset] of Object.entries(DATASETS)) {
    progress[key] = getDatasetProgress(learningSourcePath, key, dataset);
  }

  // 全体統計
  const allSteps = Object.values(progress).reduce(
    (acc, ds) => acc + ds.progress.total,
    0
  );
  const completedAllSteps = Object.values(progress).reduce(
    (acc, ds) => acc + ds.progress.completed,
    0
  );
  const overallPercent = Math.round((completedAllSteps / allSteps) * 100);

  res.json({
    learningSourceDir: learningSourceDir,
    datasets: progress,
    overall: {
      completed: completedAllSteps,
      total: allSteps,
      percent: overallPercent,
      completedDatasets: Object.values(progress).filter(
        (ds) => ds.status === 'completed'
      ).length,
      totalDatasets: Object.keys(progress).length,
    },
  });
});

/**
 * GET /api/dataset-progress/file-preview
 * ファイルの冒頭部分をプレビュー取得
 * 注意: このルートは /:datasetKey より前に定義する必要があります
 */
router.get('/file-preview', (req, res) => {
  const { filePath, datasetKey } = req.query;

  console.log('[file-preview] Received request');
  console.log('[file-preview] req.query:', req.query);
  console.log('[file-preview] filePath:', filePath);
  console.log('[file-preview] datasetKey:', datasetKey);

  if (!filePath) {
    return res.status(400).json({
      error: 'filePath query parameter is required',
    });
  }

  const projectRoot = path.resolve(__dirname, '..', '..');
  let fullPath;

  // datasetKeyが指定されている場合は、learning_source_dir配下のパスとして解決
  if (datasetKey) {
    const learningSourceDir = process.env.LEARNING_SOURCE_DIR;
    if (!learningSourceDir) {
      return res.status(500).json({
        error: 'LEARNING_SOURCE_DIR environment variable is not set',
      });
    }

    const dataset = DATASETS[datasetKey];
    if (!dataset) {
      console.log('[file-preview] Dataset not found:', datasetKey);
      console.log('[file-preview] Available:', Object.keys(DATASETS));
      return res.status(404).json({
        error: 'Dataset not found',
        availableDatasets: Object.keys(DATASETS),
      });
    }

    const learningSourcePath = path.join(projectRoot, learningSourceDir);
    const datasetPath = path.join(learningSourcePath, dataset.baseDir);
    fullPath = path.join(datasetPath, filePath);
  } else {
    // datasetKeyがない場合は、プロジェクトルートからの相対パスとして解決
    fullPath = path.join(projectRoot, filePath);
  }

  console.log('[file-preview] fullPath:', fullPath);
  console.log('[file-preview] exists:', checkExists(fullPath));

  if (!checkExists(fullPath)) {
    return res.status(404).json({
      error: 'ファイルが見つかりません',
      message: `指定されたファイルが存在しないか、アクセスできません: ${path.basename(filePath)}`,
      path: filePath,
      fullPath: fullPath,
    });
  }

  try {
    const stats = fs.statSync(fullPath);
    
    if (stats.isDirectory()) {
      return res.status(400).json({
        error: 'ディレクトリです',
        message: '指定されたパスはディレクトリのため、プレビューできません',
      });
    }

    if (stats.size > 100 * 1024 * 1024) {
      return res.status(400).json({
        error: 'ファイルサイズが大きすぎます',
        message: `ファイルサイズ ${formatFileSize(stats.size)} は大きすぎてプレビューできません（上限: 100MB）`,
        size: stats.size,
        sizeFormatted: formatFileSize(stats.size),
      });
    }

    const ext = path.extname(fullPath).toLowerCase();
    const textExtensions = ['.txt', '.json', '.csv', '.tsv', '.py', '.js', '.md', 
                           '.log', '.yaml', '.yml', '.xml', '.html', '.css', 
                           '.sh', '.bash', '.sql', '.r', '.java', '.cpp', '.c'];
    
    if (!textExtensions.includes(ext)) {
      return res.status(400).json({
        error: 'サポートされていないファイル形式',
        message: `${ext || '(拡張子なし)'} 形式のファイルはテキストプレビューに対応していません`,
        extension: ext,
      });
    }

    const maxBytes = 50 * 1024;
    const buffer = Buffer.alloc(maxBytes);
    const fd = fs.openSync(fullPath, 'r');
    const bytesRead = fs.readSync(fd, buffer, 0, maxBytes, 0);
    fs.closeSync(fd);

    let content = buffer.toString('utf8', 0, bytesRead);
    
    const lines = content.split('\n');
    if (lines.length > 100) {
      content = lines.slice(0, 100).join('\n');
      content += '\n\n... (truncated)';
    }

    res.json({
      fileName: path.basename(fullPath),
      filePath: filePath,
      size: stats.size,
      sizeFormatted: formatFileSize(stats.size),
      extension: ext,
      content: content,
      truncated: bytesRead === maxBytes || lines.length > 100,
      linesShown: Math.min(lines.length, 100),
    });
  } catch (err) {
    console.error('Error reading file for preview:', err);
    return res.status(500).json({
      error: 'ファイルの読み込みに失敗しました',
      message: `ファイルの読み込み中にエラーが発生しました: ${err.message}`,
      details: err.code || 'UNKNOWN_ERROR',
    });
  }
});

/**
 * GET /api/dataset-progress/:datasetKey
 * 特定のデータセットの詳細な進捗情報を取得
 */
router.get('/:datasetKey', (req, res) => {
  const { datasetKey } = req.params;
  const learningSourceDir = process.env.LEARNING_SOURCE_DIR;

  if (!learningSourceDir) {
    return res.status(500).json({
      error: 'LEARNING_SOURCE_DIR environment variable is not set',
    });
  }

  const dataset = DATASETS[datasetKey];
  if (!dataset) {
    return res.status(404).json({
      error: 'Dataset not found',
      availableDatasets: Object.keys(DATASETS),
    });
  }

  const projectRoot = path.resolve(__dirname, '..', '..');
  const learningSourcePath = path.join(projectRoot, learningSourceDir);

  if (!checkExists(learningSourcePath)) {
    return res.status(404).json({
      error: 'Learning source directory not found',
      path: learningSourcePath,
    });
  }

  const progress = getDatasetProgress(learningSourcePath, datasetKey, dataset);

  res.json(progress);
});

/**
 * ディレクトリ内のファイル一覧を再帰的に取得
 */
function getFilesRecursively(dirPath, baseDir = '', maxDepth = 5, currentDepth = 0) {
  const files = [];

  if (currentDepth >= maxDepth) return files;

  try {
    if (!fs.existsSync(dirPath)) return files;

    const items = fs.readdirSync(dirPath);

    for (const item of items) {
      // 隠しファイルとキャッシュディレクトリをスキップ
      if (item.startsWith('.') || item === '__pycache__' || item === 'node_modules') {
        continue;
      }

      const fullPath = path.join(dirPath, item);
      const relativePath = path.join(baseDir, item);

      try {
        const stats = fs.statSync(fullPath);

        if (stats.isDirectory()) {
          // ディレクトリの場合は再帰的に探索
          const subFiles = getFilesRecursively(fullPath, relativePath, maxDepth, currentDepth + 1);
          files.push(...subFiles);
        } else if (stats.isFile()) {
          // ファイル情報を追加
          files.push({
            name: item,
            path: relativePath,
            size: stats.size,
            modified: stats.mtime.toISOString(),
            type: path.extname(item).substring(1) || 'file',
          });
        }
      } catch (err) {
        console.error(`Error processing ${fullPath}:`, err.message);
      }
    }
  } catch (err) {
    console.error(`Error reading directory ${dirPath}:`, err.message);
  }

  return files;
}

/**
 * ファイルサイズを人間が読みやすい形式に変換
 */
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * GET /api/dataset-progress/:datasetKey/step/:stepId/files
 * 特定のステップで生成されたファイル一覧を取得
 */
router.get('/:datasetKey/step/:stepId/files', (req, res) => {
  const { datasetKey, stepId } = req.params;
  const learningSourceDir = process.env.LEARNING_SOURCE_DIR;

  if (!learningSourceDir) {
    return res.status(500).json({
      error: 'LEARNING_SOURCE_DIR environment variable is not set',
    });
  }

  const dataset = DATASETS[datasetKey];
  if (!dataset) {
    return res.status(404).json({
      error: 'Dataset not found',
      availableDatasets: Object.keys(DATASETS),
    });
  }

  const step = dataset.steps.find(s => s.id === stepId);
  if (!step) {
    return res.status(404).json({
      error: 'Step not found',
      availableSteps: dataset.steps.map(s => s.id),
    });
  }

  const projectRoot = path.resolve(__dirname, '..', '..');
  const learningSourcePath = path.join(projectRoot, learningSourceDir);
  const datasetPath = path.join(learningSourcePath, dataset.baseDir);

  if (!checkExists(datasetPath)) {
    return res.status(404).json({
      error: 'Dataset directory not found',
      path: datasetPath,
    });
  }

  const files = [];
  let totalSize = 0;

  // outputDirsからファイルを収集
  if (step.outputDirs) {
    for (const dir of step.outputDirs) {
      const dirPath = dir === '.' ? datasetPath : path.join(datasetPath, dir);
      console.log(`[step-files] Checking dir: ${dir}, dirPath: ${dirPath}, exists: ${checkExists(dirPath)}`);
      if (checkExists(dirPath)) {
        const dirFiles = getFilesRecursively(dirPath, dir === '.' ? '' : dir, 3);
        console.log(`[step-files] Found ${dirFiles.length} files in ${dir}`);
        if (dirFiles.length > 0) {
          console.log(`[step-files] Sample file paths:`, dirFiles.slice(0, 3).map(f => f.path));
        }
        files.push(...dirFiles);
      }
    }
  }

  // outputFilesからファイルを収集
  if (step.outputFiles) {
    for (const file of step.outputFiles) {
      const filePath = path.join(datasetPath, file);
      if (checkExists(filePath)) {
        try {
          const stats = fs.statSync(filePath);
          if (stats.isFile()) {
            files.push({
              name: path.basename(file),
              path: file,
              size: stats.size,
              modified: stats.mtime.toISOString(),
              type: path.extname(file).substring(1) || 'file',
            });
          }
        } catch (err) {
          console.error(`Error reading file ${filePath}:`, err.message);
        }
      }
    }
  }

  // ファイルサイズをフォーマット
  const filesWithFormattedSize = files.map(file => {
    totalSize += file.size;
    return {
      ...file,
      sizeFormatted: formatFileSize(file.size),
    };
  });

  // ファイルタイプ別に集計
  const filesByType = {};
  filesWithFormattedSize.forEach(file => {
    if (!filesByType[file.type]) {
      filesByType[file.type] = [];
    }
    filesByType[file.type].push(file);
  });

  res.json({
    datasetKey: datasetKey,
    datasetName: dataset.name,
    stepId: stepId,
    stepName: step.name,
    files: filesWithFormattedSize,
    filesByType: Object.keys(filesByType).map(type => ({
      type: type,
      count: filesByType[type].length,
      files: filesByType[type],
    })),
    summary: {
      totalFiles: filesWithFormattedSize.length,
      totalSize: totalSize,
      totalSizeFormatted: formatFileSize(totalSize),
      fileTypes: Object.keys(filesByType).length,
    },
  });
});

/**
 * GET /api/dataset-progress/:datasetKey/output/:outputType/files
 * 特定の出力ファイル（可視化など）の情報を取得
 */
router.get('/:datasetKey/output/:outputType/files', (req, res) => {
  const { datasetKey, outputType } = req.params;
  const learningSourceDir = process.env.LEARNING_SOURCE_DIR;

  if (!learningSourceDir) {
    return res.status(500).json({
      error: 'LEARNING_SOURCE_DIR environment variable is not set',
    });
  }

  const dataset = DATASETS[datasetKey];
  if (!dataset) {
    return res.status(404).json({
      error: 'Dataset not found',
      availableDatasets: Object.keys(DATASETS),
    });
  }

  if (!dataset.outputs[outputType]) {
    return res.status(404).json({
      error: 'Output type not found',
      availableOutputs: Object.keys(dataset.outputs),
    });
  }

  const projectRoot = path.resolve(__dirname, '..', '..');
  const outputPath = path.join(projectRoot, dataset.outputs[outputType]);

  if (!checkExists(outputPath)) {
    return res.status(404).json({
      error: 'Output file not found',
      path: outputPath,
    });
  }

  try {
    const stats = fs.statSync(outputPath);
    const file = {
      name: path.basename(outputPath),
      path: dataset.outputs[outputType],
      size: stats.size,
      sizeFormatted: formatFileSize(stats.size),
      modified: stats.mtime.toISOString(),
      type: path.extname(outputPath).substring(1) || 'file',
      category: 'output',
      outputType: outputType,
    };

    res.json({
      datasetKey: datasetKey,
      datasetName: dataset.name,
      outputType: outputType,
      file: file,
    });
  } catch (err) {
    return res.status(500).json({
      error: 'Failed to read output file',
      message: err.message,
    });
  }
});

/**
 * GET /api/dataset-progress/:datasetKey/files
 * 特定のデータセットで生成されたファイル一覧を取得（廃止予定）
 */
router.get('/:datasetKey/files', (req, res) => {
  const { datasetKey } = req.params;
  const learningSourceDir = process.env.LEARNING_SOURCE_DIR;

  if (!learningSourceDir) {
    return res.status(500).json({
      error: 'LEARNING_SOURCE_DIR environment variable is not set',
    });
  }

  const dataset = DATASETS[datasetKey];
  if (!dataset) {
    return res.status(404).json({
      error: 'Dataset not found',
      availableDatasets: Object.keys(DATASETS),
    });
  }

  const projectRoot = path.resolve(__dirname, '..', '..');
  const learningSourcePath = path.join(projectRoot, learningSourceDir);
  const datasetPath = path.join(learningSourcePath, dataset.baseDir);

  if (!checkExists(datasetPath)) {
    return res.status(404).json({
      error: 'Dataset directory not found',
      path: datasetPath,
    });
  }

  // データセットディレクトリ内のファイル一覧を取得
  const files = getFilesRecursively(datasetPath);

  // ファイルサイズを人間が読みやすい形式に変換
  const filesWithFormattedSize = files.map(file => ({
    ...file,
    sizeFormatted: formatFileSize(file.size),
  }));

  // ファイルタイプ別に集計
  const filesByType = {};
  let totalSize = 0;

  filesWithFormattedSize.forEach(file => {
    if (!filesByType[file.type]) {
      filesByType[file.type] = [];
    }
    filesByType[file.type].push(file);
    totalSize += file.size;
  });

  // 出力ファイル（画像など）も追加
  const outputFiles = [];
  if (dataset.outputs.plot) {
    const plotPath = path.join(projectRoot, dataset.outputs.plot);
    if (checkExists(plotPath)) {
      try {
        const stats = fs.statSync(plotPath);
        outputFiles.push({
          name: path.basename(plotPath),
          path: dataset.outputs.plot,
          size: stats.size,
          sizeFormatted: formatFileSize(stats.size),
          modified: stats.mtime.toISOString(),
          type: 'png',
          category: 'visualization',
        });
        totalSize += stats.size;
      } catch (err) {
        console.error('Error reading plot file:', err.message);
      }
    }
  }

  if (dataset.outputs.scaffoldPlot) {
    const scaffoldPlotPath = path.join(projectRoot, dataset.outputs.scaffoldPlot);
    if (checkExists(scaffoldPlotPath)) {
      try {
        const stats = fs.statSync(scaffoldPlotPath);
        outputFiles.push({
          name: path.basename(scaffoldPlotPath),
          path: dataset.outputs.scaffoldPlot,
          size: stats.size,
          sizeFormatted: formatFileSize(stats.size),
          modified: stats.mtime.toISOString(),
          type: 'png',
          category: 'visualization',
        });
        totalSize += stats.size;
      } catch (err) {
        console.error('Error reading scaffold plot file:', err.message);
      }
    }
  }

  res.json({
    datasetKey: datasetKey,
    datasetName: dataset.name,
    baseDir: dataset.baseDir,
    files: filesWithFormattedSize,
    outputFiles: outputFiles,
    filesByType: Object.keys(filesByType).map(type => ({
      type: type,
      count: filesByType[type].length,
      files: filesByType[type],
    })),
    summary: {
      totalFiles: filesWithFormattedSize.length + outputFiles.length,
      totalSize: totalSize,
      totalSizeFormatted: formatFileSize(totalSize),
      fileTypes: Object.keys(filesByType).length,
    },
  });
});

module.exports = router;
