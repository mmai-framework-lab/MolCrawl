const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');

// Get LEARNING_SOURCE_DIR from environment (required)
const LEARNING_SOURCE_DIR = process.env.LEARNING_SOURCE_DIR;
if (!LEARNING_SOURCE_DIR) {
    throw new Error('LEARNING_SOURCE_DIR environment variable is required');
}
const MODEL_BASE_DIR = path.join(__dirname, '..', '..', LEARNING_SOURCE_DIR);

/**
 * BERT training configurations for different datasets
 */
const BERT_CONFIGS = {
    compounds: {
        name: 'Compounds',
        sizes: ['small', 'medium', 'large'],
        outputDirs: {
            small: 'compounds/bert-output/compounds-small',
            medium: 'compounds/bert-output/compounds-medium',
            large: 'compounds/bert-output/compounds-large',
        },
    },
    genome_sequence: {
        name: 'Genome Sequence',
        sizes: ['small', 'medium', 'large'],
        outputDirs: {
            small: 'genome_sequence/bert-output/genome_sequence-small',
            medium: 'genome_sequence/bert-output/genome_sequence-medium',
            large: 'genome_sequence/bert-output/genome_sequence-large',
        },
    },
    protein_sequence: {
        name: 'Protein Sequence',
        sizes: ['small', 'medium', 'large'],
        outputDirs: {
            small: 'protein_sequence/bert-output/protein_sequence-small',
            medium: 'protein_sequence/bert-output/protein_sequence-medium',
            large: 'protein_sequence/bert-output/protein_sequence-large',
        },
    },
    rna: {
        name: 'RNA',
        sizes: ['small', 'medium', 'large'],
        outputDirs: {
            small: 'rna/bert-output/rna-small',
            medium: 'rna/bert-output/rna-medium',
            large: 'rna/bert-output/rna-large',
        },
    },
    molecule_nl: {
        name: 'Molecule NL',
        sizes: ['small', 'medium', 'large'],
        outputDirs: {
            small: 'molecule_nl/bert-output/molecule_nl-small',
            medium: 'molecule_nl/bert-output/molecule_nl-medium',
            large: 'molecule_nl/bert-output/molecule_nl-large',
        },
    },
};

/**
 * Find all HuggingFace checkpoint directories (checkpoint-{step}/)
 */
function findCheckpointDirectories(outputDir) {
    try {
        if (!fs.existsSync(outputDir)) {
            return [];
        }

        const entries = fs.readdirSync(outputDir, { withFileTypes: true });
        const checkpoints = entries
            .filter(entry => entry.isDirectory() && entry.name.startsWith('checkpoint-'))
            .map(entry => {
                const step = parseInt(entry.name.split('-')[1]);
                return {
                    name: entry.name,
                    step: step,
                    path: path.join(outputDir, entry.name),
                };
            })
            .filter(cp => !isNaN(cp.step))
            .sort((a, b) => b.step - a.step); // Sort by step descending

        return checkpoints;
    } catch (error) {
        console.error('Error finding checkpoint directories:', error);
        return [];
    }
}

/**
 * Read BERT checkpoint metadata (config.json)
 */
function readBERTConfigMetadata(checkpointPath) {
    try {
        const configPath = path.join(checkpointPath, 'config.json');
        if (!fs.existsSync(configPath)) {
            console.error(`Config file not found: ${configPath}`);
            return null;
        }

        const configData = JSON.parse(fs.readFileSync(configPath, 'utf8'));

        return {
            num_hidden_layers: configData.num_hidden_layers || 0,
            num_attention_heads: configData.num_attention_heads || 0,
            hidden_size: configData.hidden_size || 0,
            intermediate_size: configData.intermediate_size || 0,
            vocab_size: configData.vocab_size || 0,
            max_position_embeddings: configData.max_position_embeddings || 0,
            model_type: configData.model_type || 'bert',
            architectures: configData.architectures || [],
        };
    } catch (error) {
        console.error('Error reading BERT config metadata:', error);
        console.error('Checkpoint path:', checkpointPath);
        console.error('Error details:', error.message);
        return null;
    }
}

/**
 * Read training state from trainer_state.json
 * Note: HuggingFace trainer may write NaN values which are not valid JSON
 */
function readTrainerState(checkpointPath) {
    try {
        const statePath = path.join(checkpointPath, 'trainer_state.json');
        if (!fs.existsSync(statePath)) {
            console.error(`Trainer state file not found: ${statePath}`);
            return null;
        }

        // Read file as text first
        let jsonText = fs.readFileSync(statePath, 'utf8');

        // Replace NaN, Infinity, -Infinity with null (they are not valid JSON)
        jsonText = jsonText.replace(/:\s*NaN\s*([,\}])/g, ': null$1');
        jsonText = jsonText.replace(/:\s*Infinity\s*([,\}])/g, ': null$1');
        jsonText = jsonText.replace(/:\s*-Infinity\s*([,\}])/g, ': null$1');

        const stateData = JSON.parse(jsonText);

        // Get the latest log entry
        const logHistory = stateData.log_history || [];
        let latestTrainLoss = 0.0;
        let latestEvalLoss = null; // Use null to distinguish from 0.0
        let latestLearningRate = 0.0;

        // Parse log history for latest values
        for (let i = logHistory.length - 1; i >= 0; i--) {
            const entry = logHistory[i];
            if (entry.loss !== undefined && entry.loss !== null && !isNaN(entry.loss) && latestTrainLoss === 0.0) {
                latestTrainLoss = entry.loss;
            }
            if (entry.eval_loss !== undefined && latestEvalLoss === null) {
                // Accept null values for eval_loss (they can occur during training or be replaced from NaN)
                latestEvalLoss = entry.eval_loss;
            }
            if (entry.learning_rate !== undefined && entry.learning_rate !== null && !isNaN(entry.learning_rate) && latestLearningRate === 0.0) {
                latestLearningRate = entry.learning_rate;
            }
            // Break when we have found all values we're looking for
            if (latestTrainLoss !== 0.0 && latestEvalLoss !== null && latestLearningRate !== 0.0) {
                break;
            }
        }

        return {
            global_step: stateData.global_step || 0,
            epoch: stateData.epoch || 0,
            train_loss: latestTrainLoss,
            eval_loss: latestEvalLoss === null || isNaN(latestEvalLoss) ? 0.0 : latestEvalLoss,
            learning_rate: latestLearningRate,
            best_metric: stateData.best_metric,
            best_model_checkpoint: stateData.best_model_checkpoint,
        };
    } catch (error) {
        console.error('Error reading trainer state:', error);
        console.error('Checkpoint path:', checkpointPath);
        console.error('Error details:', error.message);
        return null;
    }
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
 * Calculate BERT model size in parameters
 */
function calculateBERTModelSize(modelArgs) {
    const { num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, vocab_size } = modelArgs;

    // Embedding layer: vocab_size * hidden_size + position embeddings + token type embeddings
    const embedding = vocab_size * hidden_size + modelArgs.max_position_embeddings * hidden_size + 2 * hidden_size;

    // Transformer blocks
    const attentionParams = num_hidden_layers * (
        4 * hidden_size * hidden_size + // Q, K, V, O projections
        4 * hidden_size // biases
    );

    const ffnParams = num_hidden_layers * (
        2 * hidden_size * intermediate_size + // Two linear layers
        hidden_size + intermediate_size // biases
    );

    const layerNormParams = num_hidden_layers * 2 * 2 * hidden_size; // 2 layer norms per block, each with scale and shift

    // Output layer (for masked language modeling)
    const output = vocab_size * hidden_size + vocab_size;

    const total = embedding + attentionParams + ffnParams + layerNormParams + output;
    return Math.round(total / 1e6); // in millions
}

/**
 * Get training status for a specific dataset and model size
 */
async function getModelStatus(dataset, size) {
    const config = BERT_CONFIGS[dataset];
    if (!config || !config.outputDirs[size]) {
        return null;
    }

    const outputDir = path.join(MODEL_BASE_DIR, config.outputDirs[size]);

    // Find HuggingFace format checkpoints
    const checkpoints = findCheckpointDirectories(outputDir);

    if (checkpoints.length > 0) {
        // Use the latest checkpoint
        const latestCheckpoint = checkpoints[0];

        const configData = readBERTConfigMetadata(latestCheckpoint.path);
        const trainerState = readTrainerState(latestCheckpoint.path);

        if (!configData) {
            console.error(`[BERT] Failed to read config.json for ${dataset}/${size}`);
            return {
                dataset,
                size,
                status: 'error',
                exists: true,
                error: 'Could not read checkpoint config.json',
            };
        }

        if (!trainerState) {
            console.error(`[BERT] Failed to read trainer_state.json for ${dataset}/${size}`);
            return {
                dataset,
                size,
                status: 'error',
                exists: true,
                error: 'Could not read checkpoint trainer_state.json',
            };
        }

        // Extract model args
        const modelArgs = {
            num_hidden_layers: configData.num_hidden_layers,
            num_attention_heads: configData.num_attention_heads,
            hidden_size: configData.hidden_size,
            intermediate_size: configData.intermediate_size,
            vocab_size: configData.vocab_size,
            max_position_embeddings: configData.max_position_embeddings,
        };

        const modTime = getFileModTime(latestCheckpoint.path);
        const modelSize = calculateBERTModelSize({ ...modelArgs, max_position_embeddings: configData.max_position_embeddings });

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
                global_step: trainerState.global_step,
                epoch: trainerState.epoch,
                train_loss: trainerState.train_loss,
                eval_loss: trainerState.eval_loss,
                learning_rate: trainerState.learning_rate,
                best_metric: trainerState.best_metric,
                best_model_checkpoint: trainerState.best_model_checkpoint,
                last_updated: modTime,
                model_args: modelArgs,
                model_size_m: modelSize,
                model_type: configData.model_type,
                architectures: configData.architectures,
            },
        };
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
 * GET /api/bert-training-status
 * Get training status for all datasets and model sizes
 */
router.get('/', async (req, res) => {
    try {
        const results = {};

        for (const [dataset, config] of Object.entries(BERT_CONFIGS)) {
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
        console.error('Error getting BERT training status:', error);
        res.status(500).json({
            success: false,
            error: error.message,
        });
    }
});

/**
 * GET /api/bert-training-status/:dataset
 * Get training status for a specific dataset
 */
router.get('/:dataset', async (req, res) => {
    const { dataset } = req.params;
    const config = BERT_CONFIGS[dataset];

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
 * GET /api/bert-training-status/:dataset/:size
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
