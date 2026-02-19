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
const MINICONDA_PYTHON = path.join(PROJECT_ROOT_DIR, 'miniconda', 'bin', 'python');
const PYTHON_BIN = fs.existsSync(MINICONDA_PYTHON) ? MINICONDA_PYTHON : 'python';

/**
 * Dataset-specific configurations
 */
const DATASET_CONFIGS = {
    compounds: {
        name: 'Compounds',
        start_token: '<|startofsmiles|>',
        max_length: 128,
        temperature: 1.0,
        top_k: null,
        example_prompts: [
            '<|startofsmiles|>C',
            '<|startofsmiles|>O',
            '<|startofsmiles|>N',
            '<|startofsmiles|>c1ccccc1',
        ],
    },
    genome_sequence: {
        name: 'Genome Sequence',
        start_token: '<|startoftext|>',
        max_length: 256,
        temperature: 0.8,
        top_k: 200,
        example_prompts: [
            '<|startoftext|>ATCG',
            '<|startoftext|>GCAT',
        ],
    },
    protein_sequence: {
        name: 'Protein Sequence',
        start_token: '<|startoftext|>',
        max_length: 256,
        temperature: 0.8,
        top_k: 50,
        example_prompts: [
            '<|startoftext|>MKTII',
            '<|startoftext|>MSKGE',
        ],
    },
    rna: {
        name: 'RNA',
        start_token: '<|startoftext|>',
        max_length: 256,
        temperature: 0.8,
        top_k: 100,
        example_prompts: [
            '<|startoftext|>AUCG',
            '<|startoftext|>GCAU',
        ],
    },
    molecule_nl: {
        name: 'Molecule Natural Language',
        start_token: '<|startoftext|>',
        max_length: 128,
        temperature: 0.7,
        top_k: 50,
        example_prompts: [
            '<|startoftext|>The molecule is',
            '<|startoftext|>This compound contains',
        ],
    },
};

/**
 * Get checkpoint path for a specific dataset and size
 */
function getCheckpointPath(dataset, size) {
    const sizeSuffix = size === 'xl' ? 'ex-large' : size;
    const outputDir = path.join(
        MODEL_BASE_DIR,
        dataset,
        'gpt2-output',
        `${dataset}-${sizeSuffix}`
    );

    // Check for checkpoint directories
    const checkpointDirs = fs
        .readdirSync(outputDir, { withFileTypes: true })
        .filter((dirent) => dirent.isDirectory() && /^checkpoint-\d+$/.test(dirent.name))
        .map((dirent) => ({
            name: dirent.name,
            path: path.join(outputDir, dirent.name),
            iteration: parseInt(dirent.name.split('-')[1]),
        }))
        .sort((a, b) => b.iteration - a.iteration);

    if (checkpointDirs.length > 0) {
        const checkpointPath = checkpointDirs[0].path;
        // Verify it's a real HuggingFace checkpoint by checking for required files
        const hasConfig = fs.existsSync(path.join(checkpointPath, 'config.json'));
        const hasPytorchModel = fs.existsSync(path.join(checkpointPath, 'pytorch_model.bin'));
        const hasCkpt = fs.existsSync(path.join(checkpointPath, 'ckpt.pt'));
        
        if (hasConfig && hasPytorchModel) {
            // True HuggingFace format
            return {
                path: checkpointPath,
                format: 'huggingface',
            };
        } else if (hasCkpt) {
            // Custom format with ckpt.pt in checkpoint directory
            return {
                path: checkpointPath,
                format: 'legacy',
            };
        }
    }

    // Check for legacy format in root output directory
    const legacyCheckpoint = path.join(outputDir, 'ckpt.pt');
    if (fs.existsSync(legacyCheckpoint)) {
        return {
            path: outputDir,
            format: 'legacy',
        };
    }

    return null;
}

/**
 * Run inference using Python script
 */
async function runInference(dataset, size, prompt, maxLength, temperature, topK, numSamples = 1) {
    const checkpointInfo = getCheckpointPath(dataset, size);
    if (!checkpointInfo) {
        throw new Error('No checkpoint found for this model');
    }

    const config = DATASET_CONFIGS[dataset];
    if (!config) {
        throw new Error('Invalid dataset');
    }

    // Create a temporary Python script for inference
    const inferenceScript = `
import os
import sys
import torch
from contextlib import nullcontext
import warnings

# Suppress all warnings and output to stderr
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from gpt2.model import GPT, GPTConfig

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
checkpoint_path = '${checkpointInfo.path}'
checkpoint_format = '${checkpointInfo.format}'

# Load model
if checkpoint_format == 'huggingface':
    from transformers import GPT2LMHeadModel, AutoTokenizer
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    model.eval()
    model.to(device)
else:
    ckpt_path = os.path.join(checkpoint_path, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Load dataset-specific tokenizer
    project_root = '${path.join(__dirname, '..', '..')}'
    tokenizer = None
    
    # Load domain-specific tokenizer
    dataset_type = '${dataset}'
    if dataset_type == 'compounds':
        from compounds.utils.tokenizer import CompoundsTokenizer
        vocab_file = os.path.join(project_root, 'assets', 'molecules', 'vocab.txt')
        if not os.path.exists(vocab_file):
            raise Exception(f"Vocab file not found: {vocab_file}")
        tokenizer = CompoundsTokenizer(vocab_file, 256)
        print(f"Loaded CompoundsTokenizer from: {vocab_file}", file=sys.stderr)
    elif dataset_type == 'genome_sequence':
        import sentencepiece as spm
        spm_path = os.path.join('${MODEL_BASE_DIR}', 'genome_sequence', 'spm_tokenizer.model')
        if os.path.exists(spm_path):
            tokenizer = spm.SentencePieceProcessor(model_file=spm_path)
            print(f"Loaded SentencePiece tokenizer from: {spm_path}", file=sys.stderr)
        else:
            raise Exception(f"SentencePiece model not found: {spm_path}")
    elif dataset_type == 'protein_sequence':
        from protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer
        tokenizer = create_bert_protein_tokenizer()
        print("Loaded protein sequence tokenizer", file=sys.stderr)
    elif dataset_type == 'rna':
        # RNA uses character-level tokenization
        class SimpleCharTokenizer:
            def __init__(self):
                self.vocab = {c: i for i, c in enumerate('ACGU')}
                self.vocab['<pad>'] = len(self.vocab)
                self.vocab['<eos>'] = len(self.vocab)
                self.eos_token_id = self.vocab['<eos>']
                self.pad_token_id = self.vocab['<pad>']
            def encode(self, text, return_tensors='pt'):
                ids = [self.vocab.get(c, self.pad_token_id) for c in text]
                if return_tensors == 'pt':
                    return torch.tensor([ids])
                return ids
            def decode(self, token_ids, skip_special_tokens=True):
                inv_vocab = {v: k for k, v in self.vocab.items()}
                chars = [inv_vocab.get(t, '') for t in token_ids]
                if skip_special_tokens:
                    chars = [c for c in chars if c not in ['<pad>', '<eos>']]
                return ''.join(chars)
        tokenizer = SimpleCharTokenizer()
        print("Loaded RNA character tokenizer", file=sys.stderr)
    elif dataset_type == 'molecule_nl':
        # Use MoleculeNatLangTokenizer (may fall back to MinimalTokenizer)
        from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer
        tokenizer = MoleculeNatLangTokenizer()
        print(f"Loaded MoleculeNatLangTokenizer (vocab_size: {tokenizer.vocab_size})", file=sys.stderr)
        # Check if it's using MinimalTokenizer (fallback)
        tokenizer_class_name = tokenizer.tokenizer.__class__.__name__
        print(f"Internal tokenizer type: {tokenizer_class_name}", file=sys.stderr)
    
    if tokenizer is None:
        raise Exception(f"No tokenizer available for dataset: {dataset_type}")

# Prepare input
prompt = '''${prompt.replace(/'/g, "\\'")}'''

# Encode based on tokenizer type
if dataset_type == 'genome_sequence':
    # SentencePiece returns list of integers
    input_ids_list = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
elif dataset_type == 'compounds':
    # CompoundsTokenizer has tokenize_text method
    input_ids_list = tokenizer.tokenize_text(prompt)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
elif dataset_type == 'molecule_nl':
    # MoleculeNatLangTokenizer uses internal _tokenize method
    tokenized = tokenizer._tokenize(prompt, add_eos_token=True)
    input_ids = torch.tensor([tokenized['input_ids']], dtype=torch.long).to(device)
elif dataset_type == 'rna':
    # SimpleCharTokenizer has encode method
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
elif hasattr(tokenizer, 'encode'):
    # Standard transformers tokenizer
    result = tokenizer.encode(prompt, return_tensors='pt')
    if isinstance(result, list):
        input_ids = torch.tensor([result], dtype=torch.long).to(device)
    else:
        input_ids = result.to(device)
else:
    raise Exception(f"Tokenizer does not have encode method for dataset: {dataset_type}")

# Generation parameters
max_length = ${maxLength}
temperature = ${temperature}
top_k = ${topK !== null ? topK : 'None'}
num_samples = ${numSamples}

# Generate
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

results = []
with torch.no_grad(), ctx:
    for i in range(num_samples):
        if checkpoint_format == 'huggingface':
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # Get eos_token_id and pad_token_id based on tokenizer type
            if dataset_type == 'genome_sequence':
                eos_id = tokenizer.eos_id()
                pad_id = tokenizer.pad_id() if hasattr(tokenizer, 'pad_id') else eos_id
            elif dataset_type == 'compounds':
                eos_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
                pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else eos_id
            elif dataset_type == 'rna':
                eos_id = tokenizer.eos_token_id
                pad_id = tokenizer.pad_token_id
            else:
                eos_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
                pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else eos_id
            
            output = model.generate(
                input_ids,
                max_new_tokens=max_length - input_ids.shape[1],
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_id,
                pad_token_id=pad_id
            )
        
        # Decode based on tokenizer type
        if dataset_type == 'genome_sequence':
            # SentencePiece decode
            generated_text = tokenizer.decode(output[0].tolist())
        elif dataset_type == 'compounds':
            # CompoundsTokenizer has decode method
            generated_text = tokenizer.decode(output[0].tolist())
        elif dataset_type == 'molecule_nl':
            # MoleculeNatLangTokenizer uses internal tokenizer
            # Check if MinimalTokenizer (which returns 'token_XXX' format)
            decoded = tokenizer.tokenizer.decode(output[0].tolist())
            # If it's MinimalTokenizer format, provide a warning
            if 'token_' in decoded:
                generated_text = "[Warning: Using fallback tokenizer - cannot decode to original text]\\n" + decoded[:500]
            else:
                generated_text = decoded
        elif dataset_type == 'rna':
            # SimpleCharTokenizer has decode method
            generated_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        elif hasattr(tokenizer, 'decode'):
            # Standard transformers tokenizer
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            generated_text = str(output[0].tolist())
        
        results.append(generated_text)

# Output results as JSON
import json
print(json.dumps({'success': True, 'results': results}))
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
            if (code !== 0) {
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
                reject(new Error(`Failed to parse inference output: ${stdout}\n${stderr}`));
            }
        });
    });
}

/**
 * POST /api/gpt2-inference
 * Run inference on a GPT-2 model
 */
router.post('/', async (req, res) => {
    try {
        const { dataset, size, prompt, maxLength, temperature, topK, numSamples } = req.body;

        if (!dataset || !size) {
            return res.status(400).json({
                success: false,
                error: 'Missing required parameters: dataset and size',
            });
        }

        const config = DATASET_CONFIGS[dataset];
        if (!config) {
            return res.status(400).json({
                success: false,
                error: 'Invalid dataset',
            });
        }

        // Use defaults if not provided
        const finalPrompt = prompt || config.start_token;
        const finalMaxLength = maxLength || config.max_length;
        const finalTemperature = temperature !== undefined ? temperature : config.temperature;
        const finalTopK = topK !== undefined ? topK : config.top_k;
        const finalNumSamples = numSamples || 1;

        const result = await runInference(
            dataset,
            size,
            finalPrompt,
            finalMaxLength,
            finalTemperature,
            finalTopK,
            finalNumSamples
        );

        res.json(result);
    } catch (error) {
        console.error('Error running inference:', error);
        res.status(500).json({
            success: false,
            error: error.message,
        });
    }
});

/**
 * GET /api/gpt2-inference/config/:dataset
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
