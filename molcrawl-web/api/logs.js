const fs = require('fs');
const path = require('path');
const { promisify } = require('util');

const readdir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const stat = promisify(fs.stat);

// LEARNING_SOURCE_DIRを取得
const getLearningSourcePath = () => {
    const learningSourceDir = process.env.LEARNING_SOURCE_DIR;
    if (!learningSourceDir) {
        throw new Error('LEARNING_SOURCE_DIR environment variable is not set');
    }
    return path.join(__dirname, '..', '..', learningSourceDir);
};

/**
 * 指定されたモデルディレクトリのログファイル一覧を取得
 */
const getLogsList = async (req, res) => {
    try {
        const { modelPath } = req.query;

        if (!modelPath) {
            return res.status(400).json({
                success: false,
                error: 'modelPath parameter is required'
            });
        }

        const learningSourcePath = getLearningSourcePath();
        const logsDir = path.join(learningSourcePath, modelPath, 'logs');

        // ディレクトリの存在確認
        if (!fs.existsSync(logsDir)) {
            return res.json({
                success: true,
                data: {
                    modelPath,
                    logsDir: path.relative(learningSourcePath, logsDir),
                    logs: [],
                    message: 'Logs directory does not exist'
                }
            });
        }

        // ディレクトリ内のファイル一覧を取得
        const files = await readdir(logsDir);

        // ログファイルの詳細情報を取得
        const logFiles = await Promise.all(
            files
                .filter(file => {
                    // ログファイルとして扱うファイルをフィルタリング
                    const ext = path.extname(file).toLowerCase();
                    return ext === '.log' || ext === '.txt' || ext === '.out' || ext === '.err' || file.endsWith('.log');
                })
                .map(async (file) => {
                    const filePath = path.join(logsDir, file);
                    const stats = await stat(filePath);

                    return {
                        name: file,
                        path: path.relative(learningSourcePath, filePath),
                        size: stats.size,
                        modifiedTime: stats.mtime.toISOString(),
                        createdTime: stats.birthtime.toISOString(),
                        isSymbolicLink: stats.isSymbolicLink()
                    };
                })
        );

        // 更新日時で降順にソート（新しいものが先）
        logFiles.sort((a, b) => new Date(b.modifiedTime) - new Date(a.modifiedTime));

        res.json({
            success: true,
            data: {
                modelPath,
                logsDir: path.relative(learningSourcePath, logsDir),
                logs: logFiles,
                count: logFiles.length
            }
        });

    } catch (error) {
        console.error('Error getting logs list:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
};

/**
 * 全モデルのログファイル一覧を取得
 */
const getAllLogsOverview = async (req, res) => {
    try {
        const learningSourcePath = getLearningSourcePath();

        // 主要なモデルディレクトリのリスト
        const modelDirs = [
            'compounds',
            'compounds_guacamol',
            'genome_sequence',
            'protein_sequence',
            'rna',
            'molecule_nl'
        ];

        const overview = await Promise.all(
            modelDirs.map(async (modelDir) => {
                const logsDir = path.join(learningSourcePath, modelDir, 'logs');

                if (!fs.existsSync(logsDir)) {
                    return {
                        modelPath: modelDir,
                        exists: false,
                        logCount: 0,
                        logs: []
                    };
                }

                try {
                    const files = await readdir(logsDir);
                    const logFiles = files.filter(file => {
                        const ext = path.extname(file).toLowerCase();
                        return ext === '.log' || ext === '.txt' || ext === '.out' || ext === '.err' || file.endsWith('.log');
                    });

                    // 各ログファイルの情報を取得
                    const logDetails = await Promise.all(
                        logFiles.map(async (file) => {
                            const filePath = path.join(logsDir, file);
                            const stats = await stat(filePath);

                            return {
                                name: file,
                                path: path.relative(learningSourcePath, filePath),
                                size: stats.size,
                                modifiedTime: stats.mtime.toISOString(),
                            };
                        })
                    );

                    // 更新日時で降順にソート
                    logDetails.sort((a, b) => new Date(b.modifiedTime) - new Date(a.modifiedTime));

                    return {
                        modelPath: modelDir,
                        exists: true,
                        logCount: logFiles.length,
                        logs: logDetails.slice(0, 5) // 最新5件のみ
                    };
                } catch (err) {
                    return {
                        modelPath: modelDir,
                        exists: true,
                        logCount: 0,
                        logs: [],
                        error: err.message
                    };
                }
            })
        );

        res.json({
            success: true,
            data: {
                models: overview,
                totalModels: modelDirs.length
            }
        });

    } catch (error) {
        console.error('Error getting all logs overview:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
};

/**
 * ログファイルの内容を取得
 */
const getLogContent = async (req, res) => {
    try {
        const { logPath } = req.query;

        if (!logPath) {
            return res.status(400).json({
                success: false,
                error: 'logPath parameter is required'
            });
        }

        const learningSourcePath = getLearningSourcePath();
        const fullPath = path.join(learningSourcePath, logPath);

        // セキュリティチェック：パスがLEARNING_SOURCE_DIR内にあることを確認
        const resolvedPath = path.resolve(fullPath);
        const resolvedBase = path.resolve(learningSourcePath);
        if (!resolvedPath.startsWith(resolvedBase)) {
            return res.status(403).json({
                success: false,
                error: 'Access denied: Path is outside of LEARNING_SOURCE_DIR'
            });
        }

        // ファイルの存在確認
        if (!fs.existsSync(fullPath)) {
            return res.status(404).json({
                success: false,
                error: 'Log file not found'
            });
        }

        // ファイルサイズチェック（大きすぎる場合は警告）
        const stats = await stat(fullPath);
        const maxSize = 3 * 1024 * 1024; // 3MB

        if (stats.size > maxSize) {
            // 大きなファイルの場合は最後の部分だけ読む
            const fileHandle = await fs.promises.open(fullPath, 'r');
            const buffer = Buffer.alloc(maxSize);
            const { bytesRead } = await fileHandle.read(buffer, 0, maxSize, stats.size - maxSize);
            await fileHandle.close();

            const content = buffer.toString('utf-8', 0, bytesRead);

            return res.json({
                success: true,
                data: {
                    path: logPath,
                    content: content,
                    size: stats.size,
                    truncated: true,
                    message: `File is too large (${Math.round(stats.size / 1024 / 1024)}MB). Showing last ${Math.round(maxSize / 1024 / 1024)}MB only. For better performance, consider using the tail endpoint.`
                }
            });
        }

        // 通常サイズのファイルは全体を読む
        const content = await readFile(fullPath, 'utf-8');

        res.json({
            success: true,
            data: {
                path: logPath,
                content: content,
                size: stats.size,
                truncated: false,
                modifiedTime: stats.mtime.toISOString()
            }
        });

    } catch (error) {
        console.error('Error reading log file:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
};

/**
 * ログファイルの末尾部分を取得（tail -f のように）
 */
const getTailLog = async (req, res) => {
    try {
        const { logPath, lines = 100 } = req.query;

        if (!logPath) {
            return res.status(400).json({
                success: false,
                error: 'logPath parameter is required'
            });
        }

        const learningSourcePath = getLearningSourcePath();
        const fullPath = path.join(learningSourcePath, logPath);

        // セキュリティチェック
        const resolvedPath = path.resolve(fullPath);
        const resolvedBase = path.resolve(learningSourcePath);
        if (!resolvedPath.startsWith(resolvedBase)) {
            return res.status(403).json({
                success: false,
                error: 'Access denied'
            });
        }

        if (!fs.existsSync(fullPath)) {
            return res.status(404).json({
                success: false,
                error: 'Log file not found'
            });
        }

        // ファイル全体を読んで最後のN行を取得
        const content = await readFile(fullPath, 'utf-8');
        const allLines = content.split('\n');
        const lastLines = allLines.slice(-parseInt(lines)).join('\n');

        const stats = await stat(fullPath);

        res.json({
            success: true,
            data: {
                path: logPath,
                content: lastLines,
                totalLines: allLines.length,
                displayedLines: Math.min(allLines.length, parseInt(lines)),
                size: stats.size,
                modifiedTime: stats.mtime.toISOString()
            }
        });

    } catch (error) {
        console.error('Error reading log tail:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
};

module.exports = {
    getLogsList,
    getAllLogsOverview,
    getLogContent,
    getTailLog
};
