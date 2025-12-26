const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

// Get LEARNING_SOURCE_DIR from environment (required)
const LEARNING_SOURCE_DIR = process.env.LEARNING_SOURCE_DIR;
if (!LEARNING_SOURCE_DIR) {
    throw new Error('LEARNING_SOURCE_DIR environment variable is required');
}
const MODEL_BASE_DIR = path.join(__dirname, '..', '..', LEARNING_SOURCE_DIR);

/**
 * GPT-2 training configurations for different datasets
 */
const GPT2_CONFIGS = {
    compounds: {
        name: 'Compounds',
        sizes: ['small', 'medium', 'large', 'xl'],
        outputDirs: {
            small: 'compounds/gpt2-output/compounds-small',
            medium: 'compounds/gpt2-output/compounds-medium',
            large: 'compounds/gpt2-output/compounds-large',
            xl: 'compounds/gpt2-output/compounds-ex-large',
        },
    },
    genome_sequence: {
        name: 'Genome Sequence',
        sizes: ['small', 'medium', 'large'],
        outputDirs: {
            small: 'genome_sequence/gpt2-output/genome_sequence-small',
            medium: 'genome_sequence/gpt2-output/genome_sequence-medium',
            large: 'genome_sequence/gpt2-output/genome_sequence-large',
        },
    },
    protein_sequence: {
        name: 'Protein Sequence',
        sizes: ['small', 'medium', 'large'],
        outputDirs: {
            small: 'protein_sequence/gpt2-output/protein_sequence-small',
            medium: 'protein_sequence/gpt2-output/protein_sequence-medium',
            large: 'protein_sequence/gpt2-output/protein_sequence-large',
        },
    },
    rna: {
        name: 'RNA',
        sizes: ['small', 'medium', 'large'],
        outputDirs: {
            small: 'rna/gpt2-output/rna-small',
            medium: 'rna/gpt2-output/rna-medium',
            large: 'rna/gpt2-output/rna-large',
        },
    },
    molecule_nl: {
        name: 'Molecule NL',
        sizes: ['small', 'medium', 'large'],
        outputDirs: {
            small: 'molecule_nl/gpt2-output/molecule-nl-small',
            medium: 'molecule_nl/gpt2-output/molecule-nl-medium',
            large: 'molecule_nl/gpt2-output/molecule-nl-large',
        },
    },
};

/**
 * Read checkpoint metadata using Python script
 */
async function readCheckpointMetadata(checkpointPath) {
    return new Promise((resolve, reject) => {
        const pythonScript = `
import sys
import torch
try:
    ckpt = torch.load('${checkpointPath}', map_location='cpu')
    result = {
        'iter_num': int(ckpt.get('iter_num', 0)),
        'best_val_loss': float(ckpt.get('best_val_loss', 0.0)),
        'model_args': {
            'n_layer': int(ckpt.get('model_args', {}).get('n_layer', 0)),
            'n_head': int(ckpt.get('model_args', {}).get('n_head', 0)),
            'n_embd': int(ckpt.get('model_args', {}).get('n_embd', 0)),
            'vocab_size': int(ckpt.get('model_args', {}).get('vocab_size', 0)),
            'block_size': int(ckpt.get('model_args', {}).get('block_size', 0)),
        }
    }
    import json
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({'error': str(e)}), file=sys.stderr)
    sys.exit(1)
`;

        const python = spawn('python', ['-c', pythonScript]);
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
                reject(new Error(stderr || 'Failed to read checkpoint'));
            } else {
                try {
                    const result = JSON.parse(stdout);
                    resolve(result);
                } catch (e) {
                    reject(new Error('Failed to parse checkpoint metadata'));
                }
            }
        });
    });
}

/**
 * Get file modification time
 */
function getFileModTime(filePath) {
    try {
        const stats = fs.statSync(filePath);
        return stats.mtime;
    } catch (error) {
        return null;
    }
}

/**
 * Calculate model size in parameters
 */
function calculateModelSize(modelArgs) {
    const { n_layer, n_head, n_embd, vocab_size } = modelArgs;
    // Rough estimation: embedding + transformer blocks + output
    const embedding = vocab_size * n_embd;
    const transformerBlock = n_layer * (
        4 * n_embd * n_embd + // attention
        8 * n_embd * n_embd    // FFN (4x expansion)
    );
    const output = vocab_size * n_embd;
    const total = embedding + transformerBlock + output;
    return Math.round(total / 1e6); // in millions
}

/**
 * Get training status for a specific dataset and model size
 */
async function getModelStatus(dataset, size) {
    const config = GPT2_CONFIGS[dataset];
    if (!config || !config.outputDirs[size]) {
        return null;
    }

    const outputDir = path.join(MODEL_BASE_DIR, config.outputDirs[size]);
    const checkpointPath = path.join(outputDir, 'ckpt.pt');

    if (!fs.existsSync(checkpointPath)) {
        return {
            dataset,
            size,
            status: 'not_started',
            exists: false,
        };
    }

    try {
        const metadata = await readCheckpointMetadata(checkpointPath);
        const modTime = getFileModTime(checkpointPath);
        const modelSize = calculateModelSize(metadata.model_args);

        return {
            dataset,
            size,
            status: 'training',
            exists: true,
            checkpoint: {
                path: config.outputDirs[size],
                iteration: metadata.iter_num,
                best_val_loss: metadata.best_val_loss,
                last_updated: modTime,
                model_args: metadata.model_args,
                model_size_m: modelSize,
            },
        };
    } catch (error) {
        return {
            dataset,
            size,
            status: 'error',
            exists: true,
            error: error.message,
        };
    }
}

/**
 * GET /api/gpt2-training-status
 * Get training status for all datasets and model sizes
 */
router.get('/', async (req, res) => {
    try {
        const results = {};

        for (const [dataset, config] of Object.entries(GPT2_CONFIGS)) {
            results[dataset] = {
                name: config.name,
                models: {},
            };

            for (const size of config.sizes) {
                const status = await getModelStatus(dataset, size);
                results[dataset].models[size] = status;
            }
        }

        res.json({
            success: true,
            learning_source_dir: LEARNING_SOURCE_DIR,
            data: results,
        });
    } catch (error) {
        console.error('Error getting GPT-2 training status:', error);
        res.status(500).json({
            success: false,
            error: error.message,
        });
    }
});

/**
 * GET /api/gpt2-training-status/:dataset
 * Get training status for a specific dataset
 */
router.get('/:dataset', async (req, res) => {
    const { dataset } = req.params;
    const config = GPT2_CONFIGS[dataset];

    if (!config) {
        return res.status(404).json({
            success: false,
            error: `Dataset '${dataset}' not found`,
        });
    }

    try {
        const models = {};
        for (const size of config.sizes) {
            const status = await getModelStatus(dataset, size);
            models[size] = status;
        }

        res.json({
            success: true,
            learning_source_dir: LEARNING_SOURCE_DIR,
            data: {
                dataset,
                name: config.name,
                models,
            },
        });
    } catch (error) {
        console.error(`Error getting training status for ${dataset}:`, error);
        res.status(500).json({
            success: false,
            error: error.message,
        });
    }
});

/**
 * GET /api/gpt2-training-status/:dataset/:size
 * Get training status for a specific model
 */
router.get('/:dataset/:size', async (req, res) => {
    const { dataset, size } = req.params;

    try {
        const status = await getModelStatus(dataset, size);

        if (!status) {
            return res.status(404).json({
                success: false,
                error: `Model '${dataset}/${size}' not found`,
            });
        }

        res.json({
            success: true,
            learning_source_dir: LEARNING_SOURCE_DIR,
            data: status,
        });
    } catch (error) {
        console.error(`Error getting status for ${dataset}/${size}:`, error);
        res.status(500).json({
            success: false,
            error: error.message,
        });
    }
});

module.exports = router;
