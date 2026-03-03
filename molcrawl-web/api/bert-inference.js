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
const PROJECT_ROOT_DIR = path.join(__dirname, '..', '..');
const BERT_DIR = path.join(PROJECT_ROOT_DIR, 'src', 'bert');
const MINICONDA_PYTHON = path.join(PROJECT_ROOT_DIR, 'miniconda', 'bin', 'python');
const PYTHON_BIN = fs.existsSync(MINICONDA_PYTHON) ? MINICONDA_PYTHON : 'python';

/**
 * Dataset-specific configurations for BERT masked language modeling
 */
const DATASET_CONFIGS = {
    compounds: {
        name: 'Compounds',
        mask_token: '[MASK]',
        example_prompts: [
            'C[MASK]C',
            'c1ccc[MASK]c1',
            'CC(=O)[MASK]',
            'O=C([MASK])O',
        ],
        description: 'SMILES molecular structures',
    },
    genome_sequence: {
        name: 'Genome Sequence',
        mask_token: '[MASK]',
        example_prompts: [
            'ATCG[MASK]ATCG',
            'GGCC[MASK][MASK]AAAA',
            'TATA[MASK]TATA',
        ],
        description: 'DNA sequences',
    },
    protein_sequence: {
        name: 'Protein Sequence',
        mask_token: '[MASK]',
        example_prompts: [
            'MKTII[MASK]LLLL',
            'ACDEG[MASK][MASK]KL',
            'MVHLT[MASK]EEKS',
        ],
        description: 'Protein amino acid sequences',
    },
    rna: {
        name: 'RNA',
        mask_token: '[MASK]',
        example_prompts: [
            'AUCG[MASK]AUCG',
            'GGCC[MASK][MASK]UUUU',
            'GUCA[MASK]GUCA',
        ],
        description: 'RNA sequences',
    },
    molecule_nat_lang: {
        name: 'Molecule Natural Language',
        mask_token: '[MASK]',
        example_prompts: [
            'This molecule has [MASK] atoms',
            'The compound is [MASK]',
            'It can be used for [MASK]',
        ],
        description: 'Natural language descriptions of molecules',
    },
};

/**
 * Get checkpoint path for a specific dataset
 */
function getCheckpointPath(dataset) {
    console.log(`[BERT] Looking for checkpoint for dataset: ${dataset}`);
    console.log(`[BERT] MODEL_BASE_DIR: ${MODEL_BASE_DIR}`);

    // Try different possible directory structures
    const possibleDirs = [
        path.join(MODEL_BASE_DIR, dataset, 'bert-output', `${dataset}-small`),
        path.join(MODEL_BASE_DIR, dataset, 'bert-output', dataset),
    ];

    for (const outputDir of possibleDirs) {
        console.log(`[BERT] Checking directory: ${outputDir}`);
        console.log(`[BERT] Directory exists: ${fs.existsSync(outputDir)}`);

        if (!fs.existsSync(outputDir)) {
            continue;
        }

        // Check for checkpoint directories
        const checkpointDirs = fs
            .readdirSync(outputDir, { withFileTypes: true })
            .filter((dirent) => dirent.isDirectory() && /^checkpoint-\d+$/.test(dirent.name))
            .map((dirent) => ({
                name: dirent.name,
                path: path.join(outputDir, dirent.name),
                step: parseInt(dirent.name.split('-')[1]),
            }))
            .sort((a, b) => b.step - a.step);

        if (checkpointDirs.length > 0) {
            const checkpointPath = checkpointDirs[0].path;
            console.log(`[BERT] Found checkpoint path: ${checkpointPath}`);

            // Verify it's a valid checkpoint
            const hasConfig = fs.existsSync(path.join(checkpointPath, 'config.json'));
            const hasPytorchModel = fs.existsSync(path.join(checkpointPath, 'pytorch_model.bin'));
            const hasSafetensors = fs.existsSync(path.join(checkpointPath, 'model.safetensors'));

            console.log(`[BERT] Has config.json: ${hasConfig}`);
            console.log(`[BERT] Has pytorch_model.bin: ${hasPytorchModel}`);
            console.log(`[BERT] Has model.safetensors: ${hasSafetensors}`);

            if (hasConfig && (hasPytorchModel || hasSafetensors)) {
                return {
                    path: checkpointPath,
                    format: 'huggingface',
                };
            }
        }
    }

    console.log(`[BERT] No valid checkpoint found for dataset: ${dataset}`);
    return null;
}

/**
 * Get tokenizer path for a specific dataset
 * Different datasets use different tokenizers
 */
function getTokenizerPath(dataset, checkpointPath) {
    const projectRoot = path.join(__dirname, '..', '..');

    // First, check if checkpoint has its own tokenizer
    const hasTokenizerInCheckpoint = fs.existsSync(path.join(checkpointPath, 'tokenizer_config.json')) ||
                                      fs.existsSync(path.join(checkpointPath, 'vocab.txt'));

    if (hasTokenizerInCheckpoint) {
        console.log(`[BERT] Using tokenizer from checkpoint: ${checkpointPath}`);
        return checkpointPath;
    }

    // Dataset-specific tokenizer paths
    const tokenizerPaths = {
        'compounds': path.join(projectRoot, 'assets', 'molecules', 'vocab.txt'),
        'genome_sequence': checkpointPath,  // Use tokenizer from checkpoint
        'protein_sequence': checkpointPath,  // Use tokenizer from checkpoint
        'rna': checkpointPath,  // Use tokenizer from checkpoint
        'molecule_nat_lang': checkpointPath,  // Use tokenizer from checkpoint (if available)
    };

    const tokenizerPath = tokenizerPaths[dataset];
    if (tokenizerPath && fs.existsSync(tokenizerPath)) {
        console.log(`[BERT] Using dataset-specific tokenizer: ${tokenizerPath}`);
        return tokenizerPath;
    }

    // Fallback to custom_tokenizer
    const customTokenizerPath = path.join(projectRoot, 'custom_tokenizer');
    if (fs.existsSync(customTokenizerPath)) {
        console.log(`[BERT] Using fallback custom_tokenizer: ${customTokenizerPath}`);
        return customTokenizerPath;
    }

    // Last resort: use checkpoint path
    console.log(`[BERT] No external tokenizer found, using checkpoint path: ${checkpointPath}`);
    return checkpointPath;
}

/**
 * Run masked language modeling inference using Python script
 */
async function runInference(dataset, text, topK = 5) {
    const checkpointInfo = getCheckpointPath(dataset);
    if (!checkpointInfo) {
        throw new Error('No checkpoint found for this model');
    }

    const config = DATASET_CONFIGS[dataset];
    if (!config) {
        throw new Error('Invalid dataset');
    }

    // Get appropriate tokenizer path for this dataset
    const tokenizerPath = getTokenizerPath(dataset, checkpointInfo.path);

    // Create a temporary Python script for masked language modeling
    const inferenceScript = `
import os
import sys
import torch
import warnings
import json

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = '${checkpointInfo.path}'
dataset = '${dataset}'

try:
    from transformers import BertForMaskedLM, BertTokenizer

    # Load model
    model = BertForMaskedLM.from_pretrained(checkpoint_path)
    model.eval()
    model.to(device)

    # Load tokenizer based on dataset type
    if dataset == 'compounds':
        # Compounds uses vocab.txt directly
        vocab_file = '${tokenizerPath}'
        if not vocab_file.endswith('.txt'):
            vocab_file = os.path.join('${tokenizerPath}', 'vocab.txt')
        tokenizer = BertTokenizer(vocab_file=vocab_file)
    else:
        # genome_sequence, protein_sequence, rna use SentencePiece-based tokenizer
        # Reconstruct tokenizer from SentencePiece model
        import sentencepiece as spm
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from transformers import PreTrainedTokenizerFast

        # Find SentencePiece model file
        dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
        sp_model_path = os.path.join(dataset_dir, 'spm_tokenizer.model')

        if os.path.exists(sp_model_path):
            # Reconstruct tokenizer using the same method as training
            sp = spm.SentencePieceProcessor(model_file=sp_model_path)
            vocab_size = sp.get_piece_size()
            vocab = [sp.id_to_piece(i) for i in range(vocab_size)]

            # Create BPE tokenizer with UNK token in vocabulary
            from tokenizers.models import WordLevel

            # Build vocabulary dict with special tokens
            vocab_dict = {}
            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

            # Add special tokens first
            for i, token in enumerate(special_tokens):
                vocab_dict[token] = i

            # Add SentencePiece vocabulary
            offset = len(special_tokens)
            for i, token in enumerate(vocab):
                if token not in vocab_dict:
                    vocab_dict[token] = i + offset

            # Create WordLevel tokenizer with the vocabulary
            tmp_tokenizer = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="[UNK]"))

            tmp_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tmp_tokenizer)
            tmp_tokenizer.unk_token = "[UNK]"
            tmp_tokenizer.sep_token = "[SEP]"
            tmp_tokenizer.pad_token = "[PAD]"
            tmp_tokenizer.cls_token = "[CLS]"
            tmp_tokenizer.mask_token = "[MASK]"

            tokenizer = tmp_tokenizer

            # Debug: verify special tokens
            print(f"[DEBUG] Tokenizer special tokens:")
            print(f"  mask_token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
            print(f"  unk_token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
            print(f"  pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
            print(f"  cls_token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
            print(f"  sep_token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
        else:
            # Fallback: try loading from checkpoint directory
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
            except:
                # Last resort: use BertTokenizer with vocab.txt if available
                vocab_file = os.path.join(checkpoint_path, 'vocab.txt')
                if os.path.exists(vocab_file):
                    tokenizer = BertTokenizer(vocab_file=vocab_file)
                else:
                    raise Exception(f"Could not load tokenizer for {dataset}. SentencePiece model not found at {sp_model_path}")

    # Input text with [MASK] tokens
    text = '''${text.replace(/'/g, "\\'")}'''
    print(f"[DEBUG] Input text: {text}")
    print(f"[DEBUG] Tokenizer mask_token: {tokenizer.mask_token}")
    print(f"[DEBUG] Tokenizer mask_token_id: {tokenizer.mask_token_id}")

    # Tokenize
    inputs = tokenizer(text, return_tensors='pt').to(device)

    # Find mask token positions
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_positions) == 0:
        print(json.dumps({
            'success': False,
            'error': 'No [MASK] token found in input text'
        }))
        sys.exit(0)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Get predictions for each mask position
    results = []
    for mask_pos in mask_positions:
        mask_predictions = predictions[0, mask_pos]
        top_k = ${topK}
        top_k_indices = torch.topk(mask_predictions, top_k).indices.tolist()
        top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
        top_k_scores = torch.softmax(mask_predictions, dim=0)[top_k_indices].tolist()

        results.append({
            'position': mask_pos.item(),
            'predictions': [
                {'token': token, 'score': score}
                for token, score in zip(top_k_tokens, top_k_scores)
            ]
        })

    # Create filled examples with top prediction
    filled_texts = []
    for i, mask_result in enumerate(results):
        # Use the top prediction for each mask
        filled_input_ids = inputs['input_ids'].clone()
        top_token_id = torch.topk(predictions[0, mask_result['position']], 1).indices[0]
        filled_input_ids[0, mask_result['position']] = top_token_id
        filled_text = tokenizer.decode(filled_input_ids[0], skip_special_tokens=True)
        filled_texts.append(filled_text)

    # Output results
    print(json.dumps({
        'success': True,
        'original_text': text,
        'mask_predictions': results,
        'filled_examples': filled_texts[:3]  # Limit to 3 examples
    }))

except Exception as e:
    import traceback
    print(json.dumps({
        'success': False,
        'error': str(e),
        'traceback': traceback.format_exc()
    }))
    sys.exit(1)
`;

    return new Promise((resolve, reject) => {
        const pythonProcess = spawn(PYTHON_BIN, ['-c', inferenceScript]);
        let stdout = '';
        let stderr = '';

        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0 && !stdout) {
                reject(new Error(`Inference failed: ${stderr}`));
                return;
            }

            try {
                // Find the last line that contains valid JSON
                const lines = stdout.trim().split('\n');
                let jsonResult = null;

                // Try to parse from the last line backwards
                for (let i = lines.length - 1; i >= 0; i--) {
                    const line = lines[i].trim();
                    if (line.startsWith('{') && line.endsWith('}')) {
                        try {
                            jsonResult = JSON.parse(line);
                            break;
                        } catch (e) {
                            continue;
                        }
                    }
                }

                if (jsonResult) {
                    resolve(jsonResult);
                } else {
                    reject(new Error(`Failed to parse inference output: ${stdout}\n${stderr}`));
                }
            } catch (err) {
                reject(new Error(`Failed to parse inference output: ${err.message}\n${stdout}\n${stderr}`));
            }
        });
    });
}

/**
 * POST /api/bert-inference
 * Run masked language modeling on a BERT model
 */
router.post('/', async (req, res) => {
    try {
        const { dataset, text, topK } = req.body;

        if (!dataset) {
            return res.status(400).json({
                success: false,
                error: 'Missing required parameter: dataset',
            });
        }

        if (!text) {
            return res.status(400).json({
                success: false,
                error: 'Missing required parameter: text',
            });
        }

        const config = DATASET_CONFIGS[dataset];
        if (!config) {
            return res.status(400).json({
                success: false,
                error: 'Invalid dataset',
            });
        }

        // Check if text contains mask token
        if (!text.includes('[MASK]')) {
            return res.status(400).json({
                success: false,
                error: 'Text must contain at least one [MASK] token',
            });
        }

        const finalTopK = topK || 5;

        const result = await runInference(dataset, text, finalTopK);

        res.json(result);
    } catch (error) {
        console.error('Error running BERT inference:', error);
        res.status(500).json({
            success: false,
            error: error.message,
        });
    }
});

/**
 * GET /api/bert-inference/config/:dataset
 * Get inference configuration for a dataset
 */
router.get('/config/:dataset', (req, res) => {
    const { dataset } = req.params;
    const config = DATASET_CONFIGS[dataset];

    if (!config) {
        return res.status(404).json({
            success: false,
            error: 'Dataset not found',
        });
    }

    res.json({
        success: true,
        config,
    });
});

module.exports = router;
