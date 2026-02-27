# Multimodal Biological LLM Framework

---

# Quick Start (Minimum Steps)

Follow these steps after cloning the repository.

---

## 0. Clone Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

---

## 1. Create Data Directory

Create the directory that will store all processed data and caches.

```bash
mkdir -p learning_source
```

---

## 2. Set Environment Variable (Required)

```bash
export LEARNING_SOURCE_DIR="$(pwd)/learning_source"
```

To make it permanent (bash example):

```bash
echo 'export LEARNING_SOURCE_DIR="$(pwd)/learning_source"' >> ~/.bashrc
```

Make sure this directory has at least 100GB of free space. Hugging Face cache will be stored inside `${LEARNING_SOURCE_DIR}/.cache/huggingface/`.

---

## 3. Create Environment (Anaconda not allowed)

```bash
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict

conda env create --name ENV_NAME --file=environment.yaml
conda activate ENV_NAME
pip install --no-build-isolation -e .
```

---

## 4. Preprocess Raw Data (Download + Tokenize)

Example: Genome Sequence

```bash
python -m src.preparation.preparation_script_genome_sequence assets/configs/genome_sequence.yaml
```

For other modalities:

```bash
python -m src.preparation.preparation_script_<task> assets/configs/<task>.yaml
```

---

## 5. Build GPT-2/BERT Training Dataset

Example: Genome Sequence

```bash
python src/genome_sequence/dataset/prepare_gpt2.py assets/configs/genome_sequence.yaml
```

For other modalities:

```bash
python src/<task>/dataset/prepare_gpt2.py assets/configs/<task>.yaml
```

---

## 6. Training

### GPT-2 (Decoder)

Small:

```bash
python gpt2/train.py gpt2/configs/<dataset>/train_gpt2_config.py
```

Medium:

```bash
python gpt2/train.py gpt2/configs/<dataset>/train_gpt2_medium_config.py
```

Large:

```bash
python gpt2/train.py gpt2/configs/<dataset>/train_gpt2_large_config.py
```

Extra-Large (XL):

```bash
python gpt2/train.py gpt2/configs/<dataset>/train_gpt2_xl_config.py
```

---

### BERT (Encoder)

```bash
python bert/main.py bert/configs/<dataset>.py
```

Make sure the GPT-2 dataset preparation step (`prepare_gpt2.py`) has been completed before BERT training, since BERT uses the same prepared dataset format.

---

### Multi-GPU (Single Node, for GPT-2 and BERT)

Both GPT-2 and BERT support Distributed Data Parallel (DDP) via `torchrun`.

Example (use GPU 0 and 2 on a 3-GPU machine):

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
<entry_script> <config_file>
```

Examples:

```bash
# GPT2
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
gpt2/train.py gpt2/configs/<dataset>/train_gpt2_config.py

# BERT
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --standalone --nproc_per_node=2 \
bert/main.py bert/configs/<dataset>.py
```

Effective batch size:

```
effective_batch_size = batch_size × gradient_accumulation_steps × num_gpus
```

Adjust `batch_size` or `gradient_accumulation_steps` accordingly to maintain reproducibility.

---

### Hardware and Practical Notes

- Large and XL models require ≥32GB GPU memory
- Multiple GPUs are recommended for faster training
- Reduce batch size if memory errors occur

---

## 7. Testing

### GPT-2

Runs automatic evaluation on the test split, including language modeling metrics such as perplexity and token-level accuracy. It can also generate sample outputs and optionally convert the checkpoint to Hugging Face format.

```bash
python gpt2/test_checkpoint.py \
  --checkpoint_path <path_to_checkpoint> \
  --domain <dataset>
```

Optional arguments:

- `--max_test_samples` (default: use full test set)
  Limit the number of test samples for faster evaluation.

- `--convert_to_hf` (default: False)
  Convert the trained checkpoint to Hugging Face format after evaluation.

---

### BERT

Runs evaluation on the test split using masked language modeling (MLM) metrics. It also supports prediction inspection and embedding extraction depending on configuration.

```bash
python bert/test_checkpoint.py \
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

# Detailed Documentation

## Repository Structure

```
assets/
bert/
gpt2/
scripts/
src/
```

- `src/`
  Core implementation of all modalities. Includes data preprocessing pipelines, dataset builders, and task-specific logic (e.g., genome, protein, RNA-seq, compounds, molecule-related natural language).

- `assets/configs/`
  YAML configuration files for preprocessing. Each modality has a corresponding config that controls download paths, tokenizer settings, and preprocessing parameters.

- `gpt2/`
  GPT-2 training code (decoder models). Contains training script (`train.py`) and model-size-specific configuration files under `gpt2/configs/<dataset>/`.

- `bert/`
  BERT training code (encoder models). The main entry point is `bert/main.py`, with dataset-specific training configs under `bert/configs/`.

- `scripts/`
  Utility scripts for dataset download or auxiliary processing steps.

This structure separates:
- Data preparation (`src/` + `assets/configs/`)
- Decoder training (`gpt2/`)
- Encoder training (`bert/`)

so preprocessing and model training remain modular and reusable.

---

### Genome Sequence

Preprocessing:

```bash
python -m src.preparation.preparation_script_genome_sequence assets/configs/genome_sequence.yaml
```

This pipeline includes:

- RefSeq download
- FASTA to raw conversion
- SentencePiece tokenizer training (or optional pretrained tokenizer)
- Conversion to parquet token files

Important config parameters:

- `path_species` (default: `assets/genome_species_list/filtered_species_refseq`)\
  Directory containing species lists used for RefSeq download.

- `num_worker` (default: `16`)\
  Number of parallel workers used for preprocessing (download is internally capped).

- `vocab_size` (default: `4096`)\
  Vocabulary size for the BPE tokenizer.

- `input_sentence_size` (default: `700000`)\
  Number of sequences sampled to train the tokenizer.

---

### Protein Sequence

Preprocessing:

```bash
python -m src.preparation.preparation_script_protein_sequence assets/configs/protein_sequence.yaml
```

Pipeline steps:

- UniRef download
- FASTA extraction
- Raw file generation
- Tokenization with ESM tokenizer
- `token_counts.pkl` generation

Key config parameters:

- `dataset` (default: `UniRef50`)\
  UniProt dataset to download (e.g., UniRef50, UniRef90, UniRef100).

- `max_lines_per_file` (default: `10**6`)\
  Number of sequences per raw/parquet file and memory loading chunk size.

- `num_worker` (default: `4`)\
  Number of workers used during download and preprocessing.

---

### RNA-seq

Preprocessing:

```bash
python -m src.preparation.preparation_script_rna assets/configs/rna.yaml
```

Pipeline steps:

- CellxGene census download
- h5ad extraction
- loom conversion
- Geneformer-style tokenization

Key config parameters:

- `census_version` (default: `2023-12-15`)\
  Version of the CellxGene census used for RNA-seq download.

- `num_worker` (default: `8`)\
  Number of parallel workers used for processing.

- `min_counts_genes` (default: `2`)\
  Minimum gene count threshold used to filter low-expression genes.

---

### Compounds

Preprocessing:

```bash
python -m src.preparation.preparation_script_compounds assets/configs/compounds.yaml
```

GPT2 preparation:

```bash
python src/compounds/dataset/prepare_gpt2.py assets/configs/compounds.yaml
```

Output contains:

- SMILES tokens
- Scaffold tokens
- Molecular properties

Key config parameters:

- `organix13_dataset` (default: `src/compounds/dataset/organix13`)\
  Directory where the raw OrganiX13 dataset is stored after download.

- `save_path` (default: `{LEARNING_SOURCE_DIR}/compounds/organix13_tokenized.parquet`)\
  Output path for the processed and tokenized parquet dataset.

- `vocab_path` (default: `assets/molecules/vocab.txt`)\
  Path to the SMILES tokenizer vocabulary file.

- `max_length` (default: `256`)\
  Maximum token length for SMILES and scaffold sequences.

---

### Molecule-related Natural Language

Download dataset:

```bash
bash src/preparation/download_smolinstruct.sh
```

Preprocess:

```bash
python -m src.preparation.preparation_script_molecule_related_nat_lang assets/configs/molecules_nl.yaml
```

Output is a HuggingFace DatasetDict with train/valid/test splits.

Key config parameters:

- `dataset` (default: `src/molecule_related_nl/assets/raw_data/osunlp/SMolInstruct`)\
  Directory where the raw SMolInstruct dataset is downloaded.

- `save_path` (default: `{LEARNING_SOURCE_DIR}/molecule_nl/molecule_related_natural_language_tokenized.parquet`)\
  Output directory for the processed and tokenized DatasetDict.

- `max_length` (default: depends on tokenizer config)\
  Maximum sequence length used during tokenization and truncation.

---
