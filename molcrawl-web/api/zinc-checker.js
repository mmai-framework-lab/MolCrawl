const fs = require('fs').promises;
const path = require('path');
const readline = require('readline');

// ファイルの行数をカウント（ヘッダー行を除く）
async function countLinesInFile(filePath) {
  try {
    const fileStream = require('fs').createReadStream(filePath);
    const rl = readline.createInterface({
      input: fileStream,
      crlfDelay: Infinity
    });

    let lineCount = 0;
    let isFirstLine = true;

    for await (const line of rl) {
      if (!isFirstLine && line.trim().length > 0) {
        lineCount++;
      }
      isFirstLine = false;
    }

    return lineCount;
  } catch (error) {
    console.error(`Error counting lines in ${filePath}:`, error);
    return 0;
  }
}

// ZINCファイルリストを取得（動的にディレクトリから取得）
async function getZincFileList(baseDir) {
  const zincFiles = [];

  try {
    // ベースディレクトリ内のディレクトリを取得
    const directories = await fs.readdir(baseDir);

    for (const dir of directories) {
      const dirPath = path.join(baseDir, dir);
      const stat = await fs.stat(dirPath);

      if (stat.isDirectory()) {
        // 各ディレクトリ内の.txtファイルを取得
        const files = await fs.readdir(dirPath);
        const txtFiles = files.filter(file => file.endsWith('.txt'));

        for (const file of txtFiles) {
          zincFiles.push(path.join(dir, file));
        }
      }
    }

    return zincFiles.sort(); // ソートして返す
  } catch (error) {
    console.error('Error getting ZINC file list:', error);
    return [];
  }
}

// ZINC20の総データ件数を取得
async function getZincDataCount(baseDir) {
  try {
    const expectedFiles = await getZincFileList(baseDir);
    const dataStats = {
      totalFiles: expectedFiles.length,
      processedFiles: 0,
      totalDataCount: 0,
      fileDataCounts: [],
      largestFiles: [],
      processingErrors: []
    };

    console.log(`Processing ${expectedFiles.length} ZINC files...`);

    for (const filePath of expectedFiles) {
      const fullPath = path.join(baseDir, filePath);
      try {
        const stats = await fs.stat(fullPath);

        if (stats.size > 0) {
          const lineCount = await countLinesInFile(fullPath);

          dataStats.processedFiles++;
          dataStats.totalDataCount += lineCount;
          dataStats.fileDataCounts.push({
            file: filePath,
            count: lineCount,
            size: stats.size
          });

          console.log(`Processed ${filePath}: ${lineCount} entries`);
        } else {
          dataStats.processingErrors.push(`${filePath}: Empty file`);
        }
      } catch (error) {
        dataStats.processingErrors.push(`${filePath}: ${error.message}`);
      }
    }

    // 最大のファイルをソートして取得
    dataStats.largestFiles = dataStats.fileDataCounts
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    // 統計計算
    if (dataStats.processedFiles > 0) {
      dataStats.averageEntriesPerFile = Math.round(dataStats.totalDataCount / dataStats.processedFiles);
    } else {
      dataStats.averageEntriesPerFile = 0;
    }

    return dataStats;
  } catch (error) {
    throw new Error(`ZINC data count failed: ${error.message}`);
  }
}

// ZINCファイルのチェック
async function checkZincFiles(baseDir) {
  try {
    const expectedFiles = await getZincFileList(baseDir);
    const results = {
      total: expectedFiles.length,
      existing: 0,
      missing: [],
      withSize: 0,
      totalSize: 0,
      filesWithSizes: []
    };

    for (const filePath of expectedFiles) {
      const fullPath = path.join(baseDir, filePath);
      try {
        const stats = await fs.stat(fullPath);
        results.existing++;

        if (stats.size > 0) {
          results.withSize++;
          results.totalSize += stats.size;
          results.filesWithSizes.push({
            path: filePath,
            size: stats.size,
            modified: stats.mtime
          });
        } else {
          results.missing.push(filePath + ' (empty file)');
        }
      } catch (error) {
        results.missing.push(filePath);
      }
    }

    // ディレクトリ別の統計を計算
    const dirStats = {};
    expectedFiles.forEach(filePath => {
      const dir = path.dirname(filePath);
      if (!dirStats[dir]) {
        dirStats[dir] = { total: 0, existing: 0 };
      }
      dirStats[dir].total++;
    });

    results.filesWithSizes.forEach(file => {
      const dir = path.dirname(file.path);
      if (dirStats[dir]) {
        dirStats[dir].existing++;
      }
    });

    results.directoryStats = dirStats;
    results.completionRate = (results.existing / results.total * 100).toFixed(1);
    results.sizeCompletionRate = (results.withSize / results.total * 100).toFixed(1);

    return results;
  } catch (error) {
    throw new Error(`ZINC file check failed: ${error.message}`);
  }
}

module.exports = {
  getZincFileList,
  checkZincFiles,
  getZincDataCount,
  countLinesInFile
};
