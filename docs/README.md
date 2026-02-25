# Environment Setup

## Required Environment Variables

Before running any scripts or applications, you must set the `LEARNING_SOURCE_DIR` environment variable:

```bash
# Set the environment variable
export LEARNING_SOURCE_DIR="learning_source"
```

**Important**: All Python scripts in this project require the `LEARNING_SOURCE_DIR` environment variable to be set. If not set, scripts will exit with an error message.

To make this permanent, add the export command to your shell configuration file:

```bash
# For bash users
echo 'export LEARNING_SOURCE_DIR="learning_source"' >> ~/.bashrc

# For zsh users
echo 'export LEARNING_SOURCE_DIR="learning_source"' >> ~/.zshrc
```

**Cache Configuration**: Hugging Face cache directories are automatically configured within `{LEARNING_SOURCE_DIR}/.cache/huggingface/` to avoid filling up the root partition. For example, if `LEARNING_SOURCE_DIR` is set to `learning_source`, the default cache directory will be `learning_source/.cache/huggingface/`. The cache location is determined by the `LEARNING_SOURCE_DIR` environment variable, so changing `LEARNING_SOURCE_DIR` will also change where the cache is stored. The detailed configuration can be found in `src/config/env.sh`.

Ensure your `LEARNING_SOURCE_DIR` points to a location with sufficient storage space (at least 100GB recommended).

## Installation

- Since Anaconda is prohibited, run the following commands first to configure conda channels:

```bash
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict
```

> **Note**: If the `defaults` channel still remains in your `.condarc` file (typically located at `~/.condarc` or `miniconda3/.condarc`), you may need to manually edit the file and remove the `defaults` entry. You can verify your channel configuration by running `conda config --show channels`.

- Create a conda environment using the environment.yaml file:

```bash
conda env create --name riken-fm --file=environment.yaml
```

- Activate the environment:

```bash
conda activate riken-fm
```

- Install the package: `pip install --no-build-isolation -e .`

## Usage

Use each script depending on the dataset you want to process for an LLM.

## Repository architecture

Each task is separate in it's own subdir. There are some shared functionality that you can
find in the src/utils folder.

```bash
├── assets
│   ├── configs                                 -> Configuration file for all task
│   ├── logging_config.json                     -> Logging base configuration
│   ├── genome_species_refseq                   -> List of species used for refseq downloading
│   └── molecules                               -> vocab.txt required for tokenizer
├── bert                                        -> Folder containing functionality for training and executing BERT-based models
├── gpn                                         -> Folder containing functionality for training and executing models that use MSA's
├── gpt2                                        -> Folder containing functionality for training and executing GPT2-based models
├── scripts                                     -> Global script for preprocissing the datasets.
│   └── preparation
├── src                                         -> Main package
│   ├── compounds
│   ├── genome_sequence
│   ├── molecule_related_natural_language
│   ├── protein_sequence
│   ├── rna
│   └── utils
├── setup.cfg
├── environment.yaml
├── pyproject.toml
└── README.md
```

## Recommended Specifications

Training is intensive, and middle to large models might not fit a tradicional GPU. For this we recommend that you have available GPUs with at least 32GB of capacity for the large versions of the models. Even then, reducing batch sizes to the mininmum might be required. Furthermore, having multiple GPUs would expedite training.

## Dataset preparation Scripts

All task have a global script that can be found in the `src/preparation` folder. They can be run with their corresponding configuration present in `assets/configs`.
The output will be multiple directory containing different step of the process.
If necessary it is possible to rerun only part of the full script by selecting the proper python file. See below for details.

> [!NOTE]
> If you have run all the scripts in the `src/preparation` folder successfully, you do not need to run any of these scripts again and you can skip this section completely.

In this section it is introduced the scripts that download, process, and tokenize the datasets. The scripts are contained in the `src/preparation` folder and are run independently.

### Running

Run each script following the syntax:

`python -m src.preparation.preparation_script_* assets/configs/*.yaml`,

where _ is the intended dataset, and `assets/configs/_.yaml` is the configuration file for the specific dataset.

### Logging

The script includes logging functionality, which is set up at runtime. The logs are saved to a file named `logging.log` in the directory where the processed dataset is saved.

### Output

The output of the script will be a tokenized version of the dataset, saved in the specified path `save_path` in the `assets/configs/*.yaml`. The log file will contain details of the processing steps, as well as statistics for the dataset.

### Error Handling

Any exceptions encountered during the execution are logged and re-raised, ensuring clear identification of issues during processing.

## Modalities Dataset Preparation

<!-- ------------------------------------------------------------------------------------------------------------- -->

### Compounds

#### Data Preprocessing

The script `src/preparation/preparation_script_molecules.py` downloads the necessary files and creates the OrganiX13 dataset, processes it, and tokenizes it following the [LLamol](https://github.com/Fraunhofer-SCAI/llamol) repository's tokenizer.

The resulting is a parquet file that contains a `pyarrow.Table` with rows: "smiles", "logp", "sascore", "mol_weight", "tokens", and "scaffold_tokens". "smiles", "logp", "sascore", and "mol_weight" are contextual data; while "tokens", and "scaffold_tokens" contains the tokenized SMILES string and the tokenized scaffold respectively.

Before running the script, ensure you have the following:

1. **Config File**: A YAML configuration file (available in `assets/`).
2. **Vocabulary File**: A pre-trained vocabulary for SMILES tokenization (available in `assets/molecules`). This can be modified in the Config File.

#### Configuration (Compounds)

The configuration file is used for the data preprocessing

```yaml
data_preparation:
  # Path to save the untokenized OrganiX13 dataset once is downloaded and processed by the script
  organix13_dataset: "src/compounds/dataset/organix13"

  # Path to save the processed and tokenized dataset
  save_path: "{LEARNING_SOURCE_DIR}/compounds/organix13_tokenized.parquet"

  # Path to the vocabulary
  vocab_path: "assets/molecules/vocab.txt"

  # Max length of the tokenized sequences
  max_length: 256
```

#### Running the Script (Compounds)

You can run this script with the following command:

```bash
python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml
```

To process specific datasets only:

```bash
python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml --datasets zinc20
```

For more options:

```bash
python src/preparation/preparation_script_compounds.py --help
```

#### Loading a Processed Dataset (Compounds)

In case you wish to load a dataset that was generated by this script for some other project or analysis, use:

```python
import pyarrow.parquet as pq

table = pq.read_table("path/to/{LEARNING_SOURCE_DIR}/compounds/tokenized/zinc20_tokenized.parquet")
```

<!-- ------------------------------------------------------------------------------------------------------------- -->

### Genome Sequence

You can run this script with the following command:

```bash
python -m src.preparation.preparation_script_genome_sequence assets/configs/genome_sequence.yaml
```

#### Configuration (Genome Sequence)

```yaml
data_preparation:
  # Path to a directory containing one file per species to download from refseq (see assets/genome_species_list/species for example)
  # Possible groups are archaea, bacteria, fungi, invertebrate, metagenomes, plant, protozoa, vertebrate_mammalian, vertebrate_other, viral.
  path_species: "assets/genome_species_list/filtered_species_refseq"

  # Num of parallel worker to use, note that for download the worker are capped to 3
  num_worker: 16

  max_lines_per_file: 10000

  # Size of the vocabulary of the BPE tokenizer
  vocab_size: 4096

  # Number of genome sequence to use to train the BPE tokenizer.
  # We will sample input_sentence_size randomly from input_sentence_size * 2 number of sequence.
  # So input_sentence_size * 2 / max_lines_per_file will be randomly selected for the BPE training.
  input_sentence_size: 700000
```

#### Separate Scripts (Genome Sequence)

The processing of Refseq is separate in 4 separate scripts. These scripts expect the result
of precedding directory to be present in the `output_dir` if that's not the case the scripts won't work.

- `src/genome_sequence/dataset/refseq/download_refseq.py`

  Uses `https://github.com/kblin/ncbi-genome-download` to download refseq data.
  `path_species` provide the directory containing one file per group and containing
  the name of the species to use. You can check the data base names here: <https://www.ncbi.nlm.nih.gov/datasets/genome/>

  Note that a lot of species used by scFormer where absent from refseq, so we only use the remaining ones.
  The full original species can be found in `assets/genome_species_list/species`
  And we used a filter set containing species with at least one sequence in refseq `assets/genome_species_list/filtered_species_refseq`

- `src/genome_sequence/dataset/refseq/fasta_to_raw.py`

  Generate the `raw_files` directory containing smaller raw file of size `max_lines_per_file` (total of 1.6TB)

- `src/genome_sequence/dataset/sentence_piece_tokenizer.py`

  As per specification we tried to train a BPE trainer from the raw files we generated. We provided an implementation with
  DNABERT_2, but in our experience the Hugging face implementation is too memory and time consuming.
  So our implementation uses the sentence piece library to train on a subset of the dataset.
  It's only one solution to use the pretrain Tokenizer trained by the DNABERT_2 authors.
  We left commented code in `src/genome_sequence/dataset/tokenizer.py`. In that case
  there is no need to train a new bpe tokenizer.

- `src/genome_sequence/dataset/tokenizer.py`

  Convert the raw files in parquet files of tokens in the `parquet_files` directory. The trained BPE Tokenizer was used to confirm the usage. Here we used Hugging Face library to load the raw files, but it is also possible to use a script similar to the one in the protein sequence version (not implemented here).

<!-- ------------------------------------------------------------------------------------------------------------- -->

### Molecule Related Natural Language

The script `src/preparation/preparation_script_molecule_related_nat_lang.py` preprocess and tokenizes a natural language molecule dataset. THe data is downloaded from [SMolInstruct](https://huggingface.co/datasets/osunlp/SMolInstruct). Then following the project's [GitHub repo](https://github.com/OSU-NLP-Group/LLM4Chem/tree/main), the data is preprocessed to have a "chat"-like format, following questions and answers. After this formatting of the data, the samples are tokenized and saved in a folder defined in the config file. This folder is a HuggingFace DatasetDict object which uses parquet.

The resulting file is a dictionary for 3 dataset splits: "train", "valid", and "test". Each of them have the features: "sample_id", "input", "output", "raw_input", "raw_output", "split", "task", "input_core_tag_left", "input_core_tag_right", "output_core_tag_left", "output_core_tag_right", "target", "input_text", "real_input_text", "input_ids", "attention_mask", "labels", "output_ids". From these, the most relevant are:

- input_text: contains the untokenized input
- output: contains the ground truth output
- input_ids: contains the tokenized input
- output_ids: contains the tokenized outputs

#### Configuration (Molecule NL)

```yaml
# Path to save the dataset once is downloaded (for example:)
dataset: "src/molecule_related_nl/assets/raw_data/osunlp/SMolInstruct"

# Path to save the processed and tokenized dataset
save_path: "{LEARNING_SOURCE_DIR}/molecule_nl/molecule_related_natural_language_tokenized.parquet"
```

#### Running the Script (Molecule NL)

Before running the script, ensure you have the following:

1. **Config File**: A YAML configuration file (available in `assets/`).

First, download the SMolInstruct dataset:

```bash
bash src/preparation/download_smolinstruct.sh
```

Then run the preparation script:

```bash
python -m src.preparation.preparation_script_molecule_related_nat_lang assets/configs/molecules_nl.yaml
```

##### Loading a Processed Dataset (Molecule NL)

To load a dataset that was generated by this script, use:

```python
from molecule_related_nl.utils.general import read_dataset
from datasets import DatasetDict

tokenized_dataset = DatasetDict(read_dataset("path/to/the/folder/created/by/script"))
```

<!-- ------------------------------------------------------------------------------------------------------------- -->

### Protein Sequence

You can run this script with the following command:

```bash
python -m src.preparation.preparation_script_protein_sequence assets/configs/protein_sequence.yaml
```

#### Configuration (Protein Sequence)

```yaml
# Which uniprot dataset to download must be one of the following:
# "UniprotKB_reviewed", "UniprotKB_unreviewed", "UniRef100", "UniRef90", "UniRef50", "UniParc"
dataset: "UniRef50"

# If True use md5 to check if a file needs to be downloaded again, using md5
# is very time consuming for large file. Otherwise we only check if the path exists.
use_md5: False

# Special case for Uniparc download, num of worker to use.
num_worker: 4

# Number of sequence per files for raw files and parquet. It also reflex the number
# of sequence loaded in memory during the processing of those files.
max_lines_per_file: 10**6
```

#### Outputs (Protein Sequence)

The output will be the a subdir of the output_dir containing a dataset name directory (ex: uniprot_50) containing the rest of the file:

- Archive file, for uniprot a `archive` dir will be creating containing all the files
- A fasta file extracted from the archive, for uniprot a `fasta_file` directory will be created containing all the file.
- A `raw_files` directory containing multiple file with one protein sequence per line.
- A `parquet_files` directory, containing two column parquet file tokenized sequence ("token") and the number of ("token_count")
- A `token_counts.pkl` file which contains a list of int corresponding to token_count for computing statistics of the dataset.

#### Separate Scripts (Protein Sequence)

The processing of Uniprot is separate in 3 separate scripts. These scripts expect the result
of precedding directory to be present in the `output_dir` if that's not the case the scripts won't work.

- `src/protein_sequence/dataset/uniprot/uniprot_download.py`

  Will download all uniprot files and extract them to fasta files.

- `src/protein_sequence/dataset/uniprot/fasta_to_raw.py`

  Generate the `raw_files` directory containing smaller raw file of size `max_lines_per_file`

- `src/protein_sequence/dataset/tokenizer.py`

  Convert the raw files in parquet files of tokens in the `parquet_files` directory. The ESM Tokenizer is used.
  The `token_counts.pkl` is also generated.

<!-- ------------------------------------------------------------------------------------------------------------- -->

### RNA

You can call this script with the following command:

```bash
python -m src.preparation.preparation_script_rna assets/configs/rna.yaml
```

One limitation of the rna sequence task is the fact that the sequence data are continuous and therefore
it creates a challenge for the tokenization and the model. The tokenization is made based on geneformer model. We used their code and statistics to compute our own tokenization. This is motivated by the similarity of both dataset.

#### Configuration (RNA)

```yaml
data_preparation:
  # Special case for Uniparc download, num of worker to use.
  num_worker: 8

  # Size of list of ids to give to each worker, save file will have `size_workload` number of ids in them.
  size_workload: 10000

  # Version of the CellxGene census
  census_version: "2023-12-15"

  # Filter condition to filter genes with few counts across a dataset.
  min_counts_genes: 2
```

#### Outputs (RNA)

This script will download the cellxgene dataset.
There will be multiple directory generate in the output_dir provided in the configuration

- `metadata_preparation`: containing `.tsv` files with ids to download
- `tissue_list.tsv`: List of tissue taken into account.
- `download_dir`: Raw archive file downloaded from the cellxgene database
- `extract`: h5ad file extracted from the archives
- `loom_dir`: loom files ready to be use for the tokenization
- `parquet_files`: parquet files containing tokenized gene and expression values

#### Separate Scripts (RNA)

The is 4 separate scripts for cellxgene downloading.

- `src/rna/dataset/cellxgene/script/build_list.py`

  Generate `metadata_preparation` and `tissue_list.tsv` to prepare the download.

- `src/rna/dataset/cellxgene/script/download.py`

  Actual downloading of the data in `download_dir` directory.

- `src/rna/dataset/cellxgene/script/conv.py`

  Extract h5ad files form the archived in `download_dir` and save them to the `extract` directory

- `src/rna/dataset/cellxgene/script/h5ad_to_loom.py`

  Transfer the h5ad file to loom and delete some unnecessary entries.

- `scr/rna/dataset/cellxgene/tokenization.py`
  Create the gene token vocabulary, based on geneformer code.

## Training of GPT2 model

## Usage Overview

1. Prepare your dataset subset by running `python gpt2/configs/<dataset>/prepare.py path/to/the/tokenized/dataset`

This will load the dataset, sample a subset, and create batches of the same length.
Note: the parameters `--training-set-subset-len` and `--test-set-subset-len` can be used to select the subset size. If < 1 taken as fracation of full data. If > 1 taken as number of samples.

1. Train the model by running `python gpt2/train.py path/to/corresponding/dataset/train_gpt2_config.py`

Inside each `data/<dataset>` folder, there is a file named `train_gpt2_config.py`, which contains parameters to train GPT2 in that dataset. For example: `python gpt2/train.py gpt2/configs/molecule_nl/train_gpt2_large_config.py` will train the large GPT2 model on the molecule_nl dataset.

Running this will lunch a training job, and output results in the path `out/ckpt.pt

> [!NOTE]
> If you have `torchrun`, you can run the model over multiple GPUs like:

`torchrun --standalone --nproc_per_node=4 config_file.py`

to run over 4 GPUs.

To run with DDP on 4 gpus across 2 nodes, example:

- Run on the first (master) node with example IP 123.456.123.456:
  `torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 config_file.py`
- Run on the worker node:
  `torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 config_file.py`
  (If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

1. Generate a sample from the trained checkpoint running `python gpt2/sample.py {config.py}`. This should be the same config file that you used for trainig, for example `python gpt2/sample.py gpt2/configs/molecule_nl/train_gpt2_large_config.py` for the exmaple in step 2.

## Data Preparation

In order to train a gpt2 model with one the dataset, you will need to run the `prepare_gpt2.py` script in
`{task}/dataset`.

For Protein Sequence, run the following command:

```bash
python src/protein_sequence/dataset/prepare_gpt2.py assets/configs/protein_sequence.yaml
```

For Molecule Related Natural Language, run the following command:

```bash
python src/molecule_related_nl/dataset/prepare_gpt2.py assets/configs/molecules_nl.yaml
```

For Genome Sequence, run the following command:

```bash
python src/genome_sequence/dataset/prepare_gpt2.py assets/configs/genome_sequence.yaml
```

For Compounds, run the following command:

```bash
python src/compounds/dataset/prepare_gpt2.py assets/configs/compounds.yaml
```

To create the `training_ready_hf_dataset` for the OrganiX13 compounds dataset, also run:

```bash
python src/compounds/dataset/prepare_gpt2_organix13.py assets/configs/compounds.yaml
```

For RNA, run the following command:

```bash
python src/rna/dataset/prepare_gpt2.py assets/configs/rna.yaml
```

> [!IMPORTANT]
> It us crucial that you adjust the config files in assets/configs so that for `assets/configs/genome_sequence.yaml`, `assets/configs/protein_sequence.yaml`, and `assets/configs/rna.yaml` the value `output_dir` is correctly pointing to the `output_dir` location where you saved the preprocessed data prepared in the [Modalities Dataset Preparation](#modalities-dataset-preparation) section. Make sure the same for `assets/configs/compounds.yaml`, and `assets/configs/molecules_nl.yaml`, where you should adjust the parameter `save_path` to match where your data is stored. \*

Now running these scripts will prepare the dataset in batch and make sure the context_size, here of 1024
is filled without any padding.

## Training

> [!IMPORTANT]
> Users need to adjust the config.py (e.g., dataset_dir, tokenizer_path, out_dir, tensorboard_dir, batch_size, etc.) before running train.py . A detailed list of additional parameters is provided in the [GPT2 Readme](./gpt2/README.md).

Then the training can be launch for the prepared datasets. In the path `gpt2/configs/<dataset-name>`, you will find a folder with 3 files:

1. `train_gpt2_config.py`: Config for training the small-sized version of the model,
2. `train_gpt2_medium_config.py`: Config for training the middle-sized version of the model,
3. `train_gpt2_large_config.py`: Config for training the large-sized version of the model.

Which file you pass to the training command will determine which version of the model it will train.

For Protein Sequence, the small version training can be done by running the following:

```bash
python gpt2/train.py gpt2/configs/protein_sequence/train_gpt2_config.py
```

For Molecule Related Natural Language, the small version training can can be done by running the following:

```bash
python gpt2/train.py gpt2/configs/molecule_nl/train_gpt2_config.py
```

For Genome Sequence, the small version training can can be done by running the following:

```bash
python gpt2/train.py gpt2/configs/genome_sequence/train_gpt2_config.py
```

For Compounds, the small version training can can be done by running the following:

```bash
python gpt2/train.py gpt2/configs/compounds/train_gpt2_config.py
```

For RNA, the small version training can can be done by running the following:

```bash
python gpt2/train.py gpt2/configs/rna/train_gpt2_config.py
```

This will train a model and save it in outputdir.

FOR MORE INFORMATION REGARDING THE CONFIG FILES, PLEASE REFER TO THE [GPT2 README](./gpt2/README.md).

## Sampling

In a similar way, using the same config files, you can sample some example with the following:

```bash
python gpt2/sample.py gpt2/configs/<dataset>/train_gpt2_config.py
```

For more information, check the README in the folder `gpt2`.

## Training of the BERT model

For the BERT model training we are using a custom script based on the Hugging Face Transformers library. The datasets used are the same as the ones for GPT2, since we already tokenize them we just need to randomly mask part of the tokens. So make sure to follow the section "Data Preparation" in [Training of GPT2 model](#Training of GPT2 model) before proceeding.

You can find the list of configs in `bert/configs`. Most parameter are similar to the ones in the gpt configuration. However, there is a `model_size` parameter that let you choose between small medium and large models. Note that medium correspond to BERT-large size, but we call it medium since the size in terms of parameter is close to GPT2-medium. The large model is a custom size bert model witch matches the gpt2 large size.

To run a training you can use the following command:

```bash
python bert/main.py bert/configs/<dataset>.py
```

If multiple GPUs are available, you can speed up training using `torchrun`. For example, to train on GPUs 0 and 2:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=2 bert/main.py bert/configs/compounds.py
```

Replace `CUDA_VISIBLE_DEVICES` with the indices of the GPUs you wish to use and set `--nproc_per_node` to the number of GPUs accordingly.

This will train a model and save it in outputdir.

For more information on the config files, see the [README inside the bert folder](./bert/README.md).

## Genome sequence training with mixed species GPN-MSA

There is a separate readme inside the `gpn` folder, detailling how to train and run inference for this task.

## Dataset Statistics

- **RNA** (scRNAseq expression data)
  - Size of the dataset: 221G
  - Samples: 35,822,843
  - Tokens: 90,711,564,293
  - Samples length size distribution of the full data: ![rna sample dist](assets/img/rna_tokenized_lengths_dist.png)

- **Protein Sequence** (Uniref 50)
  - Size of the dataset: 18G
  - Samples: 66,000,000
  - Tokens: 19,182,955,286
  - Samples length size distribution of the full data: ![protein sequence sample dist](assets/img/protein_sequence_tokenized_lengths_dist.png)

- **Compounds**
  - Size of the dataset: 954M
  - Samples: 13,288,710
  - SMILES tokens: 526,014,485
  - Scaffolds tokens: 350,007,581
  - Samples length size distribution SMILES: ![smiles sample dist](assets/img/compounds_tokenized_SMILES_lengths_dist.png)
  - Samples length size distribution Scaffolds: ![smiles sample dist](assets/img/compounds_tokenized_Scaffolds_lengths_dist.png)

- **Molecule-related natural language**
  - Size of the dataset: 8.9G
  - Training samples: 3,288,855
  - Training tokens: 460,447,039
  - Validation samples: 20,498
  - Validation tokens: 2,568,952
  - Test samples: 33,061
  - Test tokens: 4,514,586
  - Total samples: 3,342,414
  - Total tokens: 467,530,577
  - Samples length size distribution Training Set: ![mol nl sample dist](assets/img/molecule_nl_tokenized_train_lengths_dist.png)
  - Samples length size distribution Validation Set: ![mol nl sample dist](assets/img/molecule_nl_tokenized_validation_lengths_dist.png)
  - Samples length size distribution Test Set: ![mol nl sample dist](assets/img/molecule_nl_tokenized_test_lengths_dist.png)

- **Genome Sequence** (status: still on going)
  - Number of sequence: 248,678
  - Size of the vocabulary: 4096
  - Number of tokens: 3,025,575,847
  - Samples length size distribution of the full data: ![genome sequence sample dist](assets/img/genome_sequence_tokenized_lengths_dist.png)
