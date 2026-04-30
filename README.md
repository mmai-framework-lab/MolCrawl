# MolCrawl

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

---

### 4. Preprocess Raw Data (Download + Tokenize)

```bash
python -m molcrawl.preparation.preparation_script_genome_sequence assets/configs/genome_sequence.yaml
python -m molcrawl.preparation.preparation_script_protein_sequence assets/configs/protein_sequence.yaml
python -m molcrawl.preparation.preparation_script_rna assets/configs/rna.yaml
python -m molcrawl.preparation.preparation_script_compounds assets/configs/compounds.yaml
python -m molcrawl.preparation.preparation_script_molecule_nat_lang assets/configs/molecule_nat_lang.yaml
```

---

### 5. Build GPT-2/BERT Training Dataset

```bash
python molcrawl/data/genome_sequence/dataset/prepare_gpt2.py assets/configs/genome_sequence.yaml
python molcrawl/data/protein_sequence/dataset/prepare_gpt2.py assets/configs/protein_sequence.yaml
python molcrawl/data/rna/dataset/prepare_gpt2.py assets/configs/rna.yaml
python molcrawl/data/compounds/dataset/prepare_gpt2.py assets/configs/compounds.yaml
python molcrawl/data/molecule_nat_lang/dataset/prepare_gpt2.py assets/configs/molecule_nat_lang.yaml
```

---

### 6. Training

#### GPT-2 (Decoder)

Small:

```bash
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/genome_sequence/train_gpt2_small_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/protein_sequence/train_gpt2_small_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/rna/train_gpt2_small_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/compounds/train_gpt2_small_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/molecule_nat_lang/train_gpt2_small_config.py
```

Medium:

```bash
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/genome_sequence/train_gpt2_medium_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/protein_sequence/train_gpt2_medium_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/rna/train_gpt2_medium_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/compounds/train_gpt2_medium_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/molecule_nat_lang/train_gpt2_medium_config.py
```

Large:

```bash
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/genome_sequence/train_gpt2_large_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/protein_sequence/train_gpt2_large_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/rna/train_gpt2_large_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/compounds/train_gpt2_large_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/molecule_nat_lang/train_gpt2_large_config.py
```

Extra-Large (XL):

```bash
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/genome_sequence/train_gpt2_xl_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/protein_sequence/train_gpt2_xl_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/rna/train_gpt2_xl_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/compounds/train_gpt2_xl_config.py
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/molecule_nat_lang/train_gpt2_xl_config.py
```

---

#### BERT (Encoder)

```bash
python molcrawl/models/bert/main.py molcrawl/models/bert/configs/genome_sequence.py
python molcrawl/models/bert/main.py molcrawl/models/bert/configs/protein_sequence.py
python molcrawl/models/bert/main.py molcrawl/models/bert/configs/rna.py
python molcrawl/models/bert/main.py molcrawl/models/bert/configs/compounds.py
python molcrawl/models/bert/main.py molcrawl/models/bert/configs/molecule_nat_lang.py
```

Make sure the GPT-2 dataset preparation step (`prepare_gpt2.py`) has been completed before BERT training, since BERT uses the same prepared dataset format.

---

#### Multi-GPU (Single Node, for GPT-2 and BERT)

Both GPT-2 and BERT support Distributed Data Parallel (DDP) via `torchrun` (use GPU 0 and 2 on a 3-GPU machine).

```bash
# GPT2
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/genome_sequence/train_gpt2_small_config.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/protein_sequence/train_gpt2_small_config.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/rna/train_gpt2_small_config.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/compounds/train_gpt2_small_config.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/molecule_nat_lang/train_gpt2_small_config.py

# BERT
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/bert/main.py molcrawl/models/bert/configs/genome_sequence.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/bert/main.py molcrawl/models/bert/configs/protein_sequence.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/bert/main.py molcrawl/models/bert/configs/rna.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/bert/main.py molcrawl/models/bert/configs/compounds.py

CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
molcrawl/models/bert/main.py molcrawl/models/bert/configs/molecule_nat_lang.py
```

Effective batch size:

```text
effective_batch_size = batch_size × gradient_accumulation_steps × num_gpus
```

Adjust `batch_size` or `gradient_accumulation_steps` accordingly to maintain reproducibility.

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

- `--max_test_samples` (default: use full test set)
  Limit the number of test samples for faster evaluation.

- `--convert_to_hf` (default: False)
  Convert the trained checkpoint to Hugging Face format after evaluation.

---

#### BERT

Runs evaluation on the test split using masked language modeling (MLM) metrics. It also supports prediction inspection and embedding extraction depending on configuration.

```bash
python molcrawl/models/bert/test_checkpoint.py \
  --checkpoint_path <path_to_checkpoint> \
  --domain <dataset>
```

Optional arguments:

- `--max_test_samples` (default: use full test set)
  Limit the number of test samples for faster evaluation.

- `--extract_embeddings` (default: False)
  Export hidden representations (embeddings) from the model.

- `--show_predictions` (default: False)
  Print sample masked-token predictions for qualitative inspection.

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
