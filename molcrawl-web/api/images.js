const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');

/**
 * 指定されたモデルタイプの画像一覧を取得
 */
router.get('/:modelType', async (req, res) => {
    const { modelType } = req.params;

    // サポートされたモデルタイプの検証
    const supportedModels = ['protein_sequence', 'genome_sequence', 'compounds', 'rna', 'molecule_nat_lang'];
    if (!supportedModels.includes(modelType)) {
        return res.status(400).json({
            error: 'Invalid model type',
            supportedModels: supportedModels
        });
    }

    try {
        // Python スクリプトを実行して画像一覧を取得
        const { spawn } = require('child_process');
        const projectRoot = path.resolve(__dirname, '../..');

        const python = spawn('python3', [
            path.join(projectRoot, 'src', 'utils', 'get_model_images.py'),
            modelType
        ], {
            cwd: projectRoot,
            env: process.env
        });

        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        python.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        python.on('close', (code) => {
            if (code !== 0) {
                console.error('Python script error:', stderr);
                return res.status(500).json({
                    error: 'Failed to get images',
                    details: stderr
                });
            }

            try {
                const images = JSON.parse(stdout);

                // 各画像にWeb用の相対パスを追加
                const imagesWithWebPaths = images.map(image => ({
                    ...image,
                    webPath: `/api/images/serve/${modelType}/${encodeURIComponent(image.filename)}`,
                    thumbnailPath: `/api/images/thumbnail/${modelType}/${encodeURIComponent(image.filename)}`
                }));

                res.json({
                    modelType: modelType,
                    images: imagesWithWebPaths,
                    totalCount: imagesWithWebPaths.length
                });

            } catch (parseError) {
                console.error('JSON parse error:', parseError);
                res.status(500).json({
                    error: 'Failed to parse image data',
                    details: parseError.message
                });
            }
        });

    } catch (error) {
        console.error('Error getting model images:', error);
        res.status(500).json({
            error: 'Internal server error',
            message: error.message
        });
    }
});

/**
 * 画像ファイルを直接配信
 */
router.get('/serve/:modelType/:filename', async (req, res) => {
    const { modelType, filename } = req.params;

    try {
        const { spawn } = require('child_process');
        const projectRoot = path.resolve(__dirname, '../..');

        // Python スクリプトで画像パスを取得
        const python = spawn('python3', [
            path.join(projectRoot, 'src', 'utils', 'get_image_path.py'),
            modelType,
            filename
        ], {
            cwd: projectRoot,
            env: process.env
        });

        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        python.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        python.on('close', async (code) => {
            if (code !== 0) {
                return res.status(404).json({
                    error: 'Image not found',
                    details: stderr
                });
            }

            try {
                const result = JSON.parse(stdout);
                const imagePath = result.path;

                // ファイルの存在確認
                try {
                    await fs.access(imagePath);

                    // Content-Typeを設定してファイルを送信
                    const ext = path.extname(filename).toLowerCase();
                    let contentType = 'application/octet-stream';

                    switch (ext) {
                        case '.png':
                            contentType = 'image/png';
                            break;
                        case '.jpg':
                        case '.jpeg':
                            contentType = 'image/jpeg';
                            break;
                        case '.gif':
                            contentType = 'image/gif';
                            break;
                        case '.svg':
                            contentType = 'image/svg+xml';
                            break;
                        case '.bmp':
                            contentType = 'image/bmp';
                            break;
                    }

                    res.setHeader('Content-Type', contentType);
                    res.setHeader('Cache-Control', 'public, max-age=3600'); // 1時間キャッシュ
                    res.sendFile(path.resolve(imagePath));

                } catch (accessError) {
                    res.status(404).json({
                        error: 'Image file not found',
                        path: imagePath
                    });
                }

            } catch (parseError) {
                res.status(500).json({
                    error: 'Failed to get image path',
                    details: parseError.message
                });
            }
        });

    } catch (error) {
        console.error('Error serving image:', error);
        res.status(500).json({
            error: 'Internal server error',
            message: error.message
        });
    }
});

/**
 * サムネイル画像を生成・配信（簡易版：元画像をそのまま返す）
 */
router.get('/thumbnail/:modelType/:filename', (req, res) => {
    // 簡易実装：元画像をそのまま返す
    // 将来的にはImageMagickやSharpを使ってサムネイル生成も可能
    const { modelType, filename } = req.params;
    res.redirect(`/api/images/serve/${modelType}/${filename}`);
});

module.exports = router;
