const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');
const { checkZincFiles, getZincDataCount } = require('./zinc-checker');

// paths.pyからLEARNING_SOURCE_DIRを動的に取得
function getLearningSourcePath() {
  // 環境変数が設定されているか確認
  if (!process.env.LEARNING_SOURCE_DIR) {
    console.error('');
    console.error('❌ ERROR: LEARNING_SOURCE_DIR environment variable is not set!');
    console.error('');
    console.error('Please set the LEARNING_SOURCE_DIR environment variable before starting the server.');
    console.error('');
    console.error('Available learning_source directories:');
    const projectRoot = path.resolve(__dirname, '../..');
    try {
      const dirs = require('fs').readdirSync(projectRoot)
        .filter(name => name.startsWith('learning_source'));
      if (dirs.length > 0) {
        dirs.forEach(dir => console.error(`  - ${dir}`));
        console.error('');
        console.error('Example:');
        console.error(`  export LEARNING_SOURCE_DIR="${dirs[0]}"`);
        console.error(`  npm run dev`);
      } else {
        console.error('  (no learning_source directories found)');
      }
    } catch (e) {
      console.error('  (could not list directories)');
    }
    console.error('');
    process.exit(1);
  }

  try {
    const scriptPath = path.join(__dirname, '..', 'get_learning_source_dir.py');
    const projectRoot = path.resolve(__dirname, '../..');

    const result = execSync(`cd "${projectRoot}" && python3 "${scriptPath}"`, {
      encoding: 'utf8',
      env: process.env
    });

    const config = JSON.parse(result.trim());

    if (config.error) {
      throw new Error(config.error);
    }

    return config.absolute_path;
  } catch (error) {
    console.error('');
    console.error('❌ ERROR: Failed to get learning source path');
    console.error('');
    console.error('Details:', error.message);
    console.error('');
    console.error('Please ensure:');
    console.error('  1. LEARNING_SOURCE_DIR environment variable is set correctly');
    console.error('  2. The specified directory exists');
    console.error(`  3. Current value: ${process.env.LEARNING_SOURCE_DIR}`);
    console.error('');
    process.exit(1);
  }
}

const model_dir = getLearningSourcePath();

// デバッグ: model_dirの値を確認
console.log('directory.js loaded with model_dir:', model_dir);
console.log('model_dir exists?', require('fs').existsSync(model_dir));

// ディレクトリ存在チェック用ミドルウェア
function validateDirectoryExists(req, res, next) {
  const fsSync = require('fs');

  if (!fsSync.existsSync(model_dir)) {
    const projectRoot = path.resolve(__dirname, '../..');
    let availableDirs = [];

    try {
      availableDirs = fsSync.readdirSync(projectRoot)
        .filter(name => name.startsWith('learning_'));
    } catch (err) {
      console.error('Failed to list directories:', err);
    }

    return res.status(500).json({
      error: 'Directory Configuration Error',
      message: `LEARNING_SOURCE_DIR directory '${process.env.LEARNING_SOURCE_DIR}' does not exist`,
      details: {
        specified_dir: process.env.LEARNING_SOURCE_DIR,
        expected_path: model_dir,
        available_directories: availableDirs,
        suggestion: availableDirs.length > 0
          ? `Try setting LEARNING_SOURCE_DIR to one of: ${availableDirs.join(', ')}`
          : 'No learning_* directories found in project root'
      },
      timestamp: new Date().toISOString()
    });
  }

  next();
}


/**
 * ディレクトリの情報を取得
 */
async function getDirectoryInfo(dirPath) {
  try {
    // ディレクトリの存在チェック
    try {
      await fs.access(dirPath, fs.constants.F_OK);
    } catch (error) {
      if (error.code === 'ENOENT') {
        return {
          name: path.basename(dirPath),
          type: 'directory',
          size: 0,
          count: 0,
          path: dirPath,
          children: [],
          warning: 'ディレクトリが存在しません'
        };
      }
      throw error;
    }

    const stats = await fs.stat(dirPath);
    if (!stats.isDirectory()) {
      throw new Error('指定されたパスはディレクトリではありません');
    }

    let files;
    try {
      files = await fs.readdir(dirPath);
    } catch (error) {
      console.warn(`ディレクトリ読み取り失敗: ${dirPath}`, error.message);
      return {
        name: path.basename(dirPath),
        type: 'directory',
        size: 0,
        count: 0,
        path: dirPath,
        children: [],
        warning: 'ディレクトリの読み取りに失敗'
      };
    }

    const children = [];
    let totalSize = 0;

    for (const file of files) {
      const filePath = path.join(dirPath, file);
      try {
        const fileStats = await fs.stat(filePath);

        if (fileStats.isDirectory()) {
          // ディレクトリの場合、子要素の数を取得
          try {
            const subFiles = await fs.readdir(filePath);
            children.push({
              name: file,
              type: 'directory',
              size: 0,
              count: subFiles.length,
              path: filePath,
              children: [] // 初期状態では空
            });
          } catch (subError) {
            // 子ディレクトリの読み取りに失敗した場合でもディレクトリとして追加
            console.warn(`子ディレクトリ読み取り失敗: ${filePath}`, subError.message);
            children.push({
              name: file,
              type: 'directory',
              size: 0,
              count: 0,
              path: filePath,
              children: [],
              warning: '読み取り失敗'
            });
          }
        } else {
          // ファイルの場合
          children.push({
            name: file,
            type: 'file',
            size: fileStats.size,
            path: filePath
          });
          totalSize += fileStats.size;
        }
      } catch (error) {
        console.warn(`ファイル情報の取得に失敗: ${filePath}`, error.message);
        // エラーが発生したファイルも情報として追加（タイプ不明として）
        children.push({
          name: file,
          type: 'unknown',
          size: 0,
          path: filePath,
          warning: 'アクセス失敗'
        });
      }
    }

    // ソート：ディレクトリを先に、その後名前順
    children.sort((a, b) => {
      if (a.type !== b.type) {
        if (a.type === 'directory') return -1;
        if (b.type === 'directory') return 1;
        if (a.type === 'file') return -1;
        if (b.type === 'file') return 1;
      }
      return a.name.localeCompare(b.name);
    });

    return {
      name: path.basename(dirPath),
      type: 'directory',
      size: totalSize,
      count: children.length,
      path: dirPath,
      children
    };
  } catch (error) {
    throw new Error(`ディレクトリの読み取りに失敗: ${error.message}`);
  }
}

/**
 * ファイルサイズをフォーマット
 */
function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * APIエンドポイント: ディレクトリ構造を取得
 */
async function getDirectoryStructure(req, res) {
  const targetPath = req.query.path || model_dir;

  try {
    // 相対パスの場合はmodel_dirと結合
    let resolvedPath;
    if (path.isAbsolute(targetPath)) {
      resolvedPath = targetPath;
    } else {
      resolvedPath = path.join(model_dir, targetPath);
    }

    // パスの正規化
    const normalizedPath = path.normalize(resolvedPath);
    const basePath = path.resolve(__dirname, '../..');

    if (!normalizedPath.startsWith(basePath)) {
      return res.status(403).json({
        error: 'アクセス権限がありません',
        message: '指定されたパスにはアクセスできません',
        requestedPath: targetPath,
        resolvedPath: normalizedPath,
        basePath: basePath
      });
    }

    // ディレクトリが存在するかチェック
    try {
      await fs.access(normalizedPath, fs.constants.F_OK);
    } catch (error) {
      // ディレクトリが存在しない場合、空の構造を返す
      console.warn(`Directory not found: ${normalizedPath}`);
      return res.json({
        success: true,
        data: {
          name: path.basename(normalizedPath),
          type: 'directory',
          size: 0,
          count: 0,
          path: normalizedPath,
          children: [],
          warning: 'ディレクトリが存在しません'
        },
        timestamp: new Date().toISOString()
      });
    }

    const directoryInfo = await getDirectoryInfo(normalizedPath);

    res.json({
      success: true,
      data: directoryInfo,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('ディレクトリ取得エラー:', error);

    // エラーの種類に応じて適切なレスポンスを返す
    if (error.code === 'ENOENT') {
      return res.json({
        success: true,
        data: {
          name: path.basename(targetPath),
          type: 'directory',
          size: 0,
          count: 0,
          path: targetPath,
          children: [],
          warning: 'ディレクトリが存在しません'
        },
        timestamp: new Date().toISOString()
      });
    }

    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
}

/**
 * APIエンドポイント: 子ディレクトリを展開
 */
async function expandDirectory(req, res) {
  const targetPath = req.query.path;
  const recursive = req.query.recursive === 'true';
  const maxDepth = parseInt(req.query.maxDepth) || 3;

  if (!targetPath) {
    return res.status(400).json({
      error: 'パスが指定されていません',
      message: 'pathパラメータが必要です'
    });
  }

  try {
    // 相対パスの場合はmodel_dirと結合
    let resolvedPath;
    if (path.isAbsolute(targetPath)) {
      resolvedPath = targetPath;
    } else {
      resolvedPath = path.join(model_dir, targetPath);
    }

    // パスの正規化
    const normalizedPath = path.normalize(resolvedPath);
    const basePath = path.resolve(__dirname, '../..');

    if (!normalizedPath.startsWith(basePath)) {
      return res.status(403).json({
        error: 'アクセス権限がありません',
        message: '指定されたパスにはアクセスできません',
        requestedPath: targetPath,
        resolvedPath: normalizedPath
      });
    }

    // ディレクトリが存在するかチェック
    let stats;
    try {
      stats = await fs.stat(normalizedPath);
    } catch (error) {
      if (error.code === 'ENOENT') {
        // ディレクトリが存在しない場合、空の配列を返す
        return res.json({
          success: true,
          data: [],
          recursive: recursive,
          maxDepth: maxDepth,
          currentPath: normalizedPath,
          warning: 'ディレクトリが存在しません',
          timestamp: new Date().toISOString()
        });
      }
      throw error;
    }

    if (!stats.isDirectory()) {
      return res.status(400).json({
        error: 'ディレクトリではありません',
        message: '指定されたパスはディレクトリではありません'
      });
    }

    // 再帰的にディレクトリを読み込む関数
    const loadDirectoryRecursive = async (dirPath, currentDepth = 0) => {
      if (currentDepth >= maxDepth) {
        return [];
      }

      const files = await fs.readdir(dirPath);
      const children = [];

      for (const file of files) {
        const filePath = path.join(dirPath, file);
        try {
          const fileStats = await fs.stat(filePath);

          if (fileStats.isDirectory()) {
            const subFiles = await fs.readdir(filePath);
            const dirItem = {
              name: file,
              type: 'directory',
              size: 0,
              count: subFiles.length,
              path: filePath,
              children: []
            };

            // 再帰的モードの場合、子ディレクトリも読み込む
            if (recursive && currentDepth < maxDepth - 1) {
              dirItem.children = await loadDirectoryRecursive(filePath, currentDepth + 1);
            }

            children.push(dirItem);
          } else {
            children.push({
              name: file,
              type: 'file',
              size: fileStats.size,
              path: filePath
            });
          }
        } catch (error) {
          console.warn(`ファイル情報の取得に失敗: ${filePath}`, error.message);
        }
      }

      return children;
    };

    const children = await loadDirectoryRecursive(normalizedPath);

    // ソート
    const sortChildren = (children) => {
      children.sort((a, b) => {
        if (a.type !== b.type) {
          return a.type === 'directory' ? -1 : 1;
        }
        return a.name.localeCompare(b.name);
      });

      // 子ディレクトリも再帰的にソート
      children.forEach(child => {
        if (child.children && child.children.length > 0) {
          sortChildren(child.children);
        }
      });
    };

    sortChildren(children);

    res.json({
      success: true,
      data: children,
      recursive: recursive,
      maxDepth: maxDepth,
      currentPath: normalizedPath,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('ディレクトリ展開エラー:', error);

    // エラーの種類に応じて適切なレスポンスを返す
    if (error.code === 'ENOENT') {
      return res.json({
        success: true,
        data: [],
        recursive: recursive,
        maxDepth: maxDepth,
        currentPath: targetPath,
        warning: 'ディレクトリが存在しません',
        timestamp: new Date().toISOString()
      });
    }

    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
}

/**
 * APIエンドポイント: 完全なディレクトリツリーを取得
 */
async function getFullDirectoryTree(req, res) {
  const targetPath = req.query.path || model_dir;
  const maxDepth = parseInt(req.query.maxDepth) || 5;
  const includeFiles = req.query.includeFiles !== 'false';

  try {
    // 相対パスの場合はmodel_dirと結合
    let resolvedPath;
    if (path.isAbsolute(targetPath)) {
      resolvedPath = targetPath;
    } else {
      resolvedPath = path.join(model_dir, targetPath);
    }

    // パスの正規化
    const normalizedPath = path.normalize(resolvedPath);
    const basePath = path.resolve(__dirname, '../..');

    if (!normalizedPath.startsWith(basePath)) {
      return res.status(403).json({
        error: 'アクセス権限がありません',
        message: '指定されたパスにはアクセスできません',
        requestedPath: targetPath,
        resolvedPath: normalizedPath
      });
    }

    // ディレクトリが存在するかチェック
    try {
      await fs.access(normalizedPath, fs.constants.F_OK);
    } catch (error) {
      // ディレクトリが存在しない場合、空の構造を返す
      console.warn(`Directory not found for tree: ${normalizedPath}`);
      return res.json({
        success: true,
        data: {
          name: path.basename(normalizedPath),
          type: 'directory',
          size: 0,
          totalSize: 0,
          count: 0,
          path: normalizedPath,
          children: [],
          depth: 0,
          warning: 'ディレクトリが存在しません'
        },
        maxDepth: maxDepth,
        includeFiles: includeFiles,
        timestamp: new Date().toISOString()
      });
    }

    // 完全なツリー構造を再帰的に構築
    const buildFullTree = async (dirPath, currentDepth = 0) => {
      if (currentDepth >= maxDepth) {
        return null;
      }

      try {
        const stats = await fs.stat(dirPath);
        if (!stats.isDirectory()) {
          if (includeFiles) {
            return {
              name: path.basename(dirPath),
              type: 'file',
              size: stats.size,
              path: dirPath
            };
          }
          return null;
        }

        const files = await fs.readdir(dirPath);
        const children = [];
        let totalSize = 0;

        for (const file of files) {
          const filePath = path.join(dirPath, file);
          const child = await buildFullTree(filePath, currentDepth + 1);

          if (child) {
            children.push(child);
            if (child.type === 'file') {
              totalSize += child.size;
            } else if (child.totalSize) {
              totalSize += child.totalSize;
            }
          }
        }

        // ソート
        children.sort((a, b) => {
          if (a.type !== b.type) {
            return a.type === 'directory' ? -1 : 1;
          }
          return a.name.localeCompare(b.name);
        });

        return {
          name: path.basename(dirPath),
          type: 'directory',
          size: 0,
          totalSize: totalSize,
          count: children.length,
          path: dirPath,
          children: children,
          depth: currentDepth
        };
      } catch (error) {
        console.warn(`ディレクトリ読み取りエラー: ${dirPath}`, error.message);
        return null;
      }
    };

    const tree = await buildFullTree(normalizedPath);

    if (!tree) {
      return res.json({
        success: true,
        data: {
          name: path.basename(normalizedPath),
          type: 'directory',
          size: 0,
          totalSize: 0,
          count: 0,
          path: normalizedPath,
          children: [],
          depth: 0,
          warning: 'ディレクトリツリーの構築に失敗しました'
        },
        maxDepth: maxDepth,
        includeFiles: includeFiles,
        timestamp: new Date().toISOString()
      });
    }

    res.json({
      success: true,
      data: tree,
      maxDepth: maxDepth,
      includeFiles: includeFiles,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('完全ツリー取得エラー:', error);

    // エラーの種類に応じて適切なレスポンスを返す
    if (error.code === 'ENOENT') {
      return res.json({
        success: true,
        data: {
          name: path.basename(targetPath),
          type: 'directory',
          size: 0,
          totalSize: 0,
          count: 0,
          path: targetPath,
          children: [],
          depth: 0,
          warning: 'ディレクトリが存在しません'
        },
        maxDepth: maxDepth,
        includeFiles: includeFiles,
        timestamp: new Date().toISOString()
      });
    }

    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
}

// ZINC20データのチェック機能
async function checkZincData(req, res) {
  try {
    // compounds/dataset/organix13/zinc/zinc_complete/ のパスを構築
    const zincBasePath = path.join(model_dir, 'compounds', 'dataset', 'organix13', 'zinc', 'zinc_complete');

    console.log('Checking ZINC data at:', zincBasePath);

    // ディレクトリが存在するかチェック
    try {
      await fs.access(zincBasePath);
    } catch (error) {
      return res.json({
        success: true,
        data: {
          exists: false,
          message: 'ZINC20データディレクトリが見つかりません',
          path: zincBasePath
        }
      });
    }

    // ZINCファイルをチェック
    const checkResults = await checkZincFiles(zincBasePath);

    res.json({
      success: true,
      data: {
        exists: true,
        path: zincBasePath,
        ...checkResults,
        lastChecked: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('ZINC データチェックエラー:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
}

// ZINC20データ件数取得機能
async function getZincDataCounts(req, res) {
  try {
    // compounds/zinc20/ のパスを構築
    const zincBasePath = path.join(model_dir, 'compounds', 'zinc20');

    console.log('Counting ZINC data entries at:', zincBasePath);

    // ディレクトリが存在するかチェック
    try {
      await fs.access(zincBasePath);
    } catch (error) {
      return res.json({
        success: true,
        data: {
          exists: false,
          message: 'ZINC20データディレクトリが見つかりません',
          path: zincBasePath
        }
      });
    }

    // ZINCデータ件数をカウント
    const dataStats = await getZincDataCount(zincBasePath);

    res.json({
      success: true,
      data: {
        exists: true,
        path: zincBasePath,
        ...dataStats,
        lastCounted: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('ZINC データ件数取得エラー:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
}

module.exports = {
  getDirectoryStructure,
  expandDirectory,
  getFullDirectoryTree,
  checkZincData,
  getZincDataCounts,
  formatFileSize,
  validateDirectoryExists
};
