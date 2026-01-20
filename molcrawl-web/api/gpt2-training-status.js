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
        sizes: ['small', 'medium', 'large', 'xl'],
        outputDirs: {
            small: 'genome_sequence/gpt2-output/genome_sequence-small',
            medium: 'genome_sequence/gpt2-output/genome_sequence-medium',
            large: 'genome_sequence/gpt2-output/genome_sequence-large',
            xl: 'genome_sequence/gpt2-output/genome_sequence-ex-large',
        },
    },
    protein_sequence: {
        name: 'Protein Sequence',
        sizes: ['small', 'medium', 'large', 'xl'],
        outputDirs: {
            small: 'protein_sequence/gpt2-output/protein_sequence-small',
            medium: 'protein_sequence/gpt2-output/protein_sequence-medium',
            large: 'protein_sequence/gpt2-output/protein_sequence-large',
            xl: 'protein_sequence/gpt2-output/protein_sequence-ex-large',
        },
    },
    rna: {
        name: 'RNA',
        sizes: ['small', 'medium', 'large', 'xl'],
        outputDirs: {
            small: 'rna/gpt2-output/rna-small',
            medium: 'rna/gpt2-output/rna-medium',
            large: 'rna/gpt2-output/rna-large',
            xl: 'rna/gpt2-output/rna-ex-large',
        },
    },
    molecule_nl: {
        name: 'Molecule NL',
        sizes: ['small', 'medium', 'large', 'xl'],
        outputDirs: {
            small: 'molecule_nl/gpt2-output/molecule_nl-small',
            medium: 'molecule_nl/gpt2-output/molecule_nl-medium',
            large: 'molecule_nl/gpt2-output/molecule_nl-large',
            xl: 'molecule_nl/gpt2-output/molecule_nl-ex-large',
        },
    },
};

/**
 * Find all HuggingFace checkpoint directories (checkpoint-{step}/ or _checkpoint-{step}/)
 */
function findCheckpointDirectories(outputDir) {
    try {
        if (!fs.existsSync(outputDir)) {
            return [];
        }
        
        const entries = fs.readdirSync(outputDir, { withFileTypes: true });
        const checkpoints = entries
            .filter(entry => {
                // Support both "checkpoint-" and "_checkpoint-" prefixes
                return entry.isDirectory() && 
                       (entry.name.startsWith('checkpoint-') || entry.name.startsWith('_checkpoint-'));
            })
            .map(entry => {
                // Extract step number from both "checkpoint-123" and "_checkpoint-123"
                const parts = entry.name.split('-');
                const step = parseInt(parts[parts.length - 1]);
                const checkpointPath = path.join(outputDir, entry.name);
                
                // Check if training_args.json exists (valid checkpoint)
                const trainingArgsPath = path.join(checkpointPath, 'training_args.json');
                const hasTrainingArgs = fs.existsSync(trainingArgsPath);
                
                return {
                    name: entry.name,
                    step: step,
                    path: checkpointPath,
                    hasTrainingArgs: hasTrainingArgs,
                };
            })
            .filter(cp => !isNaN(cp.step) && cp.hasTrainingArgs) // Only include valid checkpoints with training_args.json
            .sort((a, b) => b.step - a.step); // Sort by step descending
        
        return checkpoints;
    } catch (error) {
        console.error('Error finding checkpoint directories:', error);
        return [];
    }
}

/**
 * Read HuggingFace checkpoint metadata (training_args.json)
 * Current implementation uses custom format with training_args.json
 */
function readHFCheckpointMetadata(checkpointPath) {
    try {
        const trainingArgsPath = path.join(checkpointPath, 'training_args.json');
        if (!fs.existsSync(trainingArgsPath)) {
            return null;
        }
        
        const argsData = JSON.parse(fs.readFileSync(trainingArgsPath, 'utf8'));
        
        // Extract model args from training_args.json
        const modelArgs = argsData.model_args || {};
        return {
            n_layer: modelArgs.n_layer || 0,
            n_head: modelArgs.n_head || 0,
            n_embd: modelArgs.n_embd || 0,
            vocab_size: modelArgs.vocab_size || 0,
            block_size: modelArgs.block_size || 0,
            // Also return training info
            iteration: argsData.iteration || 0,
            best_val_loss: argsData.best_val_loss || 0.0,
            learning_rate: argsData.learning_rate || 0,
            batch_size: argsData.batch_size || 0,
        };
    } catch (error) {
        console.error('Error reading HF checkpoint metadata:', error);
        return null;
    }
}

/**
 * Read training state from trainer_state.json if available
 * Note: Current implementation uses training_args.json instead
 */
function readTrainerState(checkpointPath) {
    try {
        const statePath = path.join(checkpointPath, 'trainer_state.json');
        if (fs.existsSync(statePath)) {
            const stateData = JSON.parse(fs.readFileSync(statePath, 'utf8'));
            return {
                best_val_loss: stateData.best_val_loss || 0.0,
                global_step: stateData.global_step || 0,
            };
        }
        
        // Fallback: try training_args.json
        const trainingArgsPath = path.join(checkpointPath, 'training_args.json');
        if (fs.existsSync(trainingArgsPath)) {
            const argsData = JSON.parse(fs.readFileSync(trainingArgsPath, 'utf8'));
            return {
                best_val_loss: argsData.best_val_loss || 0.0,
                global_step: argsData.iteration || 0,
            };
        }
        
        return null;
    } catch (error) {
        return null;
    }
}

/**
 * Read logging CSV file for loss information
 */
function readLoggingCSV(outputDir) {
    try {
        const loggingPath = path.join(outputDir, 'logging.csv');
        if (!fs.existsSync(loggingPath)) {
            return null;
        }
        
        const content = fs.readFileSync(loggingPath, 'utf8');
        const lines = content.trim().split('\n');
        if (lines.length < 2) {
            return null;
        }
        
        // Get last line (most recent)
        const lastLine = lines[lines.length - 1];
        const parts = lastLine.split(',').map(p => p.trim());
        
        if (parts.length >= 3) {
            return {
                iter: parseInt(parts[0]) || 0,
                train_loss: parseFloat(parts[1]) || 0.0,
                val_loss: parseFloat(parts[2]) || 0.0,
            };
        }
        
        return null;
    } catch (error) {
        return null;
    }
}

/**
 * Legacy: Read checkpoint metadata from ckpt.pt using Python script
 */
async function readLegacyCheckpointMetadata(checkpointPath) {
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
    
    // First, try to find HuggingFace format checkpoints
    const checkpoints = findCheckpointDirectories(outputDir);
    
    if (checkpoints.length > 0) {
        // Use the latest checkpoint
        const latestCheckpoint = checkpoints[0];
        const checkpointData = readHFCheckpointMetadata(latestCheckpoint.path);
        const loggingData = readLoggingCSV(outputDir);
        
        if (!checkpointData) {
            return {
                dataset,
                size,
                status: 'error',
                exists: true,
                error: 'Could not read checkpoint training_args.json',
            };
        }
        
        // Extract model args (removing training-specific fields)
        const modelArgs = {
            n_layer: checkpointData.n_layer,
            n_head: checkpointData.n_head,
            n_embd: checkpointData.n_embd,
            vocab_size: checkpointData.vocab_size,
            block_size: checkpointData.block_size,
        };
        
        const modTime = getFileModTime(latestCheckpoint.path);
        const modelSize = calculateModelSize(modelArgs);
        
        // Use data from training_args.json (most reliable) or fallback to logging.csv
        const iteration = checkpointData.iteration || latestCheckpoint.step;
        const best_val_loss = checkpointData.best_val_loss || loggingData?.val_loss || 0.0;
        const train_loss = loggingData?.train_loss || 0.0;
        const learning_rate = checkpointData.learning_rate || 0;
        const batch_size = checkpointData.batch_size || 0;
        
        return {
            dataset,
            size,
            status: 'training',
            exists: true,
            checkpoint_format: 'huggingface',
            checkpoint_count: checkpoints.length,
            checkpoint: {
                path: config.outputDirs[size],
                checkpoint_name: latestCheckpoint.name,
                iteration: iteration,
                train_loss: train_loss,
                best_val_loss: best_val_loss,
                learning_rate: learning_rate,
                batch_size: batch_size,
                last_updated: modTime,
                model_args: modelArgs,
                model_size_m: modelSize,
            },
        };
    }
    
    // Fallback: Try legacy ckpt.pt format
    const legacyCheckpointPath = path.join(outputDir, 'ckpt.pt');
    if (fs.existsSync(legacyCheckpointPath)) {
        try {
            const metadata = await readLegacyCheckpointMetadata(legacyCheckpointPath);
            const modTime = getFileModTime(legacyCheckpointPath);
            const modelSize = calculateModelSize(metadata.model_args);

            return {
                dataset,
                size,
                status: 'training',
                exists: true,
                checkpoint_format: 'legacy',
                checkpoint: {
                    path: config.outputDirs[size],
                    checkpoint_name: 'ckpt.pt',
                    iteration: metadata.iter_num,
                    train_loss: 0.0,
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
                checkpoint_format: 'legacy',
                error: error.message,
            };
        }
    }
    
    // No checkpoints found
    return {
        dataset,
        size,
        status: 'not_started',
        exists: false,
    };
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
