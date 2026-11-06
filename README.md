# ![MolCrawl](assets/logo/molcrawl_logo_horizontal.png)

[![CI Tests](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/ci-tests.yml)
[![Ruff Lint](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/ruff.yml/badge.svg)](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/ruff.yml)
[![ESLint](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/eslint.yml/badge.svg)](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/eslint.yml)
[![Compounds Validation](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/compounds-validation.yml/badge.svg)](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/compounds-validation.yml)
[![Documentation](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/documentation.yml/badge.svg)](https://github.com/mmai-framework-lab/MolCrawl/actions/workflows/documentation.yml)

MolCrawl is a pipeline framework to construct biological multimodal foundational models designed to handle chemical and life science data in a unified manner.

---

## Detailed Documentation

For comprehensive documentation including environment setup details, dataset guides, training configuration, evaluation procedures, experiment tracking, and more, see the **[Documentation Index](docs/README.md)**.

---

## Pre-trained Models

Pre-trained model checkpoints for all modalities (Genome Sequence, Protein Sequence, RNA, Compounds, and Molecule-related Natural Language) are publicly available on Hugging Face:

**[https://huggingface.co/collections/kojima-lab/molcrawl](https://huggingface.co/collections/kojima-lab/molcrawl)**

You can use these checkpoints directly for inference or as starting points for fine-tuning, without running the full data preparation and training pipeline described below.

---

## Quick Start (Minimum Steps)

Follow these steps after cloning the repository.

---

### 0. Clone Repository

```bash
git clone https://github.com/mmai-framework-lab/MolCrawl.git
cd MolCrawl
```

---

### 1. Create Data Directory

Create the directory that will store all processed data and caches.

```bash
mkdir -p learning_source
```

---

### 2. Set Environment Variable (Required)

```bash
export LEARNING_SOURCE_DIR="$(pwd)/learning_source"
```

To make it permanent (bash example):

```bash
echo 'export LEARNING_SOURCE_DIR="$(pwd)/learning_source"' >> ~/.bashrc
```

Make sure this directory has at least 100GB of free space. Hugging Face cache will be stored inside `${LEARNING_SOURCE_DIR}/.cache/huggingface/`.

Model output must live outside `LEARNING_SOURCE_DIR`: BERT training refuses to start when its output directory is inside that tree.

```bash
export MODEL_OUTPUT_ROOT="$(pwd)/learning_source_runs"
```

---

### 3. Create Environment (Anaconda not allowed)

```bash
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict

conda env create --name "molcrawl" --file=environment.yaml
conda activate molcrawl
pip install --no-build-isolation -e .
```

#### Offline / restricted-network nodes (optional)

The `molecule_nat_lang` preparation loads the GPT-2 tokenizer via
`transformers.from_pretrained("gpt2")`, which needs to reach Hugging Face at
runtime. On nodes that cannot reach `huggingface.co` this fails with a
connection error. Pre-download the tokenizer once into the directory the
workflows auto-detect (`assets/tokenizers/gpt2`):

```bash
huggingface-cli download openai-community/gpt2 \
  config.json vocab.json merges.txt tokenizer.json tokenizer_config.json \
  --local-dir assets/tokenizers/gpt2
```

The `01-/02-molecule_nat_lang` workflow scripts export `GPT2_TOKENIZER_DIR` to
this path automatically when it exists; otherwise set it manually.

---

### 4. Preprocess Raw Data (Download + Tokenize)

```bash
python molcrawl/data/genome_sequence/preparation.py assets/configs/genome_sequence.yaml
python molcrawl/data/protein_sequence/preparation.py assets/configs/protein_sequence.yaml
python molcrawl/data/rna/preparation.py assets/configs/rna.yaml
python molcrawl/data/compounds/preparation.py assets/configs/compounds.yaml
python molcrawl/data/molecule_nat_lang/preparation.py assets/configs/molecule_nat_lang_config.yaml
```

---

### 5. Build GPT-2/BERT Training Dataset

```bash
python molcrawl/data/genome_sequence/dataset/prepare_gpt2.py assets/configs/genome_sequence.yaml
python molcrawl/data/protein_sequence/dataset/prepare_gpt2.py assets/configs/protein_sequence.yaml
python molcrawl/data/rna/dataset/prepare_gpt2.py assets/configs/rna.yaml

# Compounds: packed 1024-token blocks, read by both the GPT-2 and the BERT configs
MOLCRAWL_REPO="$(pwd)" ORGANIX13_DIR="${LEARNING_SOURCE_DIR}/compounds/organix13" \
python scripts/run_phase1_2_training_ready_v4_packed.py

# Molecule NL: the configs read training_ready_hf_dataset_shuffled
PREP_OUT_SUFFIX=_shuffled \
python molcrawl/data/molecule_nat_lang/dataset/prepare_gpt2.py assets/configs/molecule_nat_lang_config.yaml
```

`molcrawl/data/compounds/dataset/prepare_gpt2.py` builds the GuacaMol benchmark dataset, not the pretraining corpus.

---

### 6. Training

#### GPT-2 (Decoder)

Small:

```bash
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/protein_sequence/gpt2_small.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/rna/gpt2_small.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/compounds/gpt2_small.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_small.py
```

Medium:

```bash
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_medium.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/protein_sequence/gpt2_medium.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/rna/gpt2_medium.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/compounds/gpt2_medium.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_medium.py
```

Large:

```bash
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_large.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/protein_sequence/gpt2_large.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/rna/gpt2_large.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/compounds/gpt2_large.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_large.py
```

Extra-Large (XL):

```bash
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_xl.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/protein_sequence/gpt2_xl.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/rna/gpt2_xl.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/compounds/gpt2_xl.py
python molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_xl.py
```

---

#### BERT (Encoder)

These configs assume 4 GPUs. The protein, compounds and molecule NL configs declare `expected_global_batch = 2560` and stop at startup on any other GPU count.

```bash
torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py molcrawl/tasks/pretrain/configs/genome_sequence/bert_small.py
torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py molcrawl/tasks/pretrain/configs/protein_sequence/bert_small.py
torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py molcrawl/tasks/pretrain/configs/rna/bert_small.py
torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py molcrawl/tasks/pretrain/configs/compounds/bert_small.py
torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py molcrawl/tasks/pretrain/configs/molecule_nat_lang/bert_small.py
```

For a smoke test on another GPU count, state the resulting batch explicitly. The run is then not comparable with the 2,560 ladder.

```bash
python molcrawl/models/bert/main.py molcrawl/tasks/pretrain/configs/compounds/bert_small.py --expected_global_batch=640
```

Make sure the dataset step above has been completed before BERT training, since BERT uses the same prepared dataset format.

---

#### Multi-GPU (Single Node, GPT-2)

GPT-2 supports Distributed Data Parallel (DDP) via `torchrun` (use GPU 0 and 2 on a 3-GPU machine). BERT also runs under `torchrun`, as shown above.

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/protein_sequence/gpt2_small.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/rna/gpt2_small.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/compounds/gpt2_small.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_small.py
```

Effective batch size:

```text
GPT-2: effective_batch_size = batch_size × gradient_accumulation_steps
BERT:  effective_batch_size = batch_size × gradient_accumulation_steps × num_gpus
```

The GPT-2 value does not depend on the GPU count; the BERT value does.

---

#### Hardware and Practical Notes

- Large and XL models require ≥32GB GPU memory
- Multiple GPUs are recommended for faster training
- Reduce batch size if memory errors occur

---

> **Note:** Steps 4–6 can also be run collectively using the workflow scripts in the `workflows/` directory.
> These scripts batch-process multiple modalities (data preparation, dataset building, and training) in a single command.
> See **[workflows/README.md](workflows/README.md)** for details.

---

### 7. Testing

#### GPT-2

Runs automatic evaluation on the test split, including language modeling metrics such as perplexity and token-level accuracy. It can also generate sample outputs and optionally convert the checkpoint to Hugging Face format.

```bash
python molcrawl/models/gpt2/test_checkpoint.py \
  --checkpoint_path <path_to_checkpoint> \
  --domain <dataset>
```

Optional arguments:

- `--max_test_samples` (default: 1000)
  Maximum number of test samples to evaluate.

- `--convert_to_hf` (default: False)
  Convert the trained checkpoint to Hugging Face format after evaluation.

---

#### BERT

Runs a checkpoint check on short sample texts: basic inference, masked language modeling (MLM), embedding generation, batch processing and performance.

```bash
python molcrawl/models/bert/test_checkpoint.py \
  --checkpoint_path <path_to_checkpoint> \
  --domain <dataset>
```

Optional arguments:

- `--dataset_path`
  Evaluation dataset path.

- `--vocab_path`
  Vocabulary file path (compounds: `vocab.txt`, genome_sequence: `spm_tokenizer.model`).

- `--test_texts`
  Sample texts to use instead of the per-domain defaults.

---

## Dataset Statistics

Genome Sequence

- Number of sequences: 248,678
- Vocabulary size: 4096
- Number of tokens: 3,025,575,847

Protein Sequence (UniRef50)

- Size of the dataset: 18G
- Samples: 66,000,000
- Tokens: 19,182,955,286

RNA-seq (scRNAseq expression data)

- Size of the dataset: 221G
- Samples: 35,822,843
- Tokens: 90,711,564,293

Compounds

- Size of the dataset: 954M
- Samples: 13,288,710
- SMILES tokens: 526,014,485
- Scaffold tokens: 350,007,581

Molecule-related Natural Language

- Size of the dataset: 8.9G
- Training samples: 3,288,855
- Training tokens: 460,447,039
- Validation samples: 20,498
- Validation tokens: 2,568,952
- Test samples: 33,061
- Test tokens: 4,514,586
- Total samples: 3,342,414
- Total tokens: 467,530,577

---
