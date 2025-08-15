const fs = require('fs').promises;
const path = require('path');
const model_dir = path.resolve(__dirname, '../../learning_source_202508');

// デバッグ: model_dirの値を確認
console.log('directory.js loaded with model_dir:', model_dir);
console.log('model_dir exists?', require('fs').existsSync(model_dir));


/**
 * ディレクトリの情報を取得
 */
async function getDirectoryInfo(dirPath) {
  try {
    const stats = await fs.stat(dirPath);
    const files = await fs.readdir(dirPath);
    
    const children = [];
    let totalSize = 0;
    
    for (const file of files) {
      const filePath = path.join(dirPath, file);
      try {
        const fileStats = await fs.stat(filePath);
        
        if (fileStats.isDirectory()) {
          // ディレクトリの場合、子要素の数を取得
          const subFiles = await fs.readdir(filePath);
          children.push({
            name: file,
            type: 'directory',
            size: 0,
            count: subFiles.length,
            path: filePath,
            children: [] // 初期状態では空
          });
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
      }
    }
    
    // ソート：ディレクトリを先に、その後名前順
    children.sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === 'directory' ? -1 : 1;
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
    
    const directoryInfo = await getDirectoryInfo(normalizedPath);
    
    res.json({
      success: true,
      data: directoryInfo,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('ディレクトリ取得エラー:', error);
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
    
    const stats = await fs.stat(normalizedPath);
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
      throw new Error('ディレクトリツリーの構築に失敗しました');
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
  formatFileSize
};
