# Installation
1. Install the package: `pip install --no-build-isolation -e .`

# Usage

Use each script depending on the dataset you want to process for an LLM.

# Repository architecture

Each task is separate in it's own subdir. There are some shared functionality that you can
find in the src/utils folder.

├── assets
│   ├── configs                                 -> Configuration file for all task
│   ├── logging_config.json                     -> Logging base configuration
│   └── molecules                               -> ???
├── scripts                                     -> Global script for all tasks.
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


# Dataset preparation

All task have a global script that can be found in the `scripts` folder. They can be run with their corresponding configuration present in `assets/configs`.
The output will be multiple directory containing different step of the process.
If necessary it is possible to rerun only part of the full script by selecting the proper python file. See below for details.


# Dataset Statistics

- <b>RNA</b> (scRNAseq expression data)

    Size of the dataset: 221G

    Samples: 35,822,843-

    Tokens: 90,711,564,293

- <b>Protein Sequence</b> (Uniref 50)

    Size of the dataset: 18G

    Samples: 66,000,000

    Tokens: 19,182,955,286

- <b>Compounds</b>:

    Size of the dataset: 954M

    Samples: 13,299,623

    SMILES Tokens: 526,014,485

    Scaffold Tokens: 350,007,581

- <b>Molecule-related natural language</b>

    Size of the dataset: 8.9G

    Samples: 3,342,414

    Tokens: 467,530,577

- <b>Genome Sequence</b> (status: not provided since imprecise)

    Size of the dataset 1.6T


<!-- ------------------------------------------------------------------------------------------------------------- -->
## Compounds

Molecules dataset uses OrganiX13. The script `scripts/preparation_script_molecules.py` processes the dataset and tokenizes it following the [LLamol](https://github.com/Fraunhofer-SCAI/llamol) repository's tokenizer. The script will preprocess the dataset and generate a parquet file containing the tokenized SMILES from the Molecules Organix13 dataset.

#### Prerequisites
Before running the script, ensure you have the following:

1. **Organix13 Dataset**: The dataset must be in Parquet format.
2. **Vocabulary File**: A pre-trained vocabulary for SMILES tokenization.
3. **Logging Configuration**: A JSON configuration file for logging.

#### Usage
To run the script, use the following command:

```bash
python <script_name>.py -o13 <organix13_dataset_path> -sp <save_path> -vp <vocab_path> [-ml <max_length>]
```

- `-o13, --organix13-dataset`: Path to the root folder of the Organix13 dataset (required).
- `-sp, --save-path`: Path to save the processed and tokenized dataset in Parquet format (required).
- `-vp, --vocab-path`: Path to the SMILES vocabulary file (required).
- `-ml, --max-length`: Maximum token length for SMILES sequences. Default is 256 (optional).

#### Example
```bash
python process_dataset.py -o13 data/organix13.parquet -sp data/tokenized_dataset.parquet -vp data/vocab.json -ml 512
```

#### Logging
The script includes logging functionality, which is set up at runtime. The logs are saved to a file named `logging.log` in the directory where the processed dataset is saved. You can customize the logging configuration by providing a JSON file (`assets/logging_config.json` by default).

#### Output
The output of the script will be a tokenized version of the OrganiX13 dataset, saved in the specified path in Parquet format. The log file will contain details of the processing steps.

#### Error Handling
Any exceptions encountered during the execution are logged and re-raised, ensuring clear identification of issues during processing.

<!-- ------------------------------------------------------------------------------------------------------------- -->

## Genome Sequence

You can run this script with the following command:

```bash
python scripts/preparation_script_genome_sequence.py assets/configs/genome_sequence.yaml
```

However, this scripts has won't be able to finish due to the size of the refseq database 4.2TB. See Separate scripts section
for more details.
To compare the dataset used in the reference paper is 32.4GB.
You can find the original pretraining dataset [here](https://github.com/MAGICS-LAB/DNABERT_2/issues/100).

### Separate scripts

The processing of Refseq is separate in 4 separate scripts. These scripts except the result
of precedding directory to be present in the `output_dir` if that's not the case the scripts won't work.

- `src/genome_sequence/dataset/refseq/download_refseq.py`

    Will download all refseq files finishing by .genomic.fna.gz from `https://ftp.ncbi.nlm.nih.gov/refseq/release/complete/` and extract them to fasta files.
    This dataset is huge, 1.3 TB for the downloaded directory, 4.2TB for the extracted files.

- `src/genome_sequence/dataset/refseq/fasta_to_raw.py`

    Generate the `raw_files` directory containing smaller raw file of size `max_lines_per_file` (total of 1.6TB)

- `src/genome_sequence/dataset/train_tokenizer.py`

    As per specification we tried to train a BPE trainer, we even follow the same code as the authors, see
    [here](https://github.com/MAGICS-LAB/DNABERT_2/issues/74) for the source code.
    However this code has a huge data usage and we could not use it without getting OOM error.
    In order to try to get it working we capped the datasize up to 10GB, but even doing so lead to a usage
    of ~350GB of RAM and the process stop undefinetelly when trying to compute the merge.

    We did manage to compute the BPE on up to 2 files, and even in this condition it took more than 12 hours to finish.
    One solution might be to go outside of Hugging Face trainer and build a more memory optimized solution for genome sequence.
    It seems that long sequence make the algorithm of hugging face particularly slow.

    From hugging face issues section, we also saw a recommendation of using Unigram rather than BPE if it is possible for your project.

- `src/genome_sequence/dataset/tokenizer.py`

    Convert the raw files in parquet files of tokens in the `parquet_files` directory. The trained BPE Tokenizer (trained on only two file) was used to confirm the usage. Here we used Hugging Face library, but it is also possible to use a script similar to the one in the protein sequence version (not implemented here).


<!-- ------------------------------------------------------------------------------------------------------------- -->

## Molecule Related Natural Language
<!-- ------------------------------------------------------------------------------------------------------------- -->

## Protein Sequence

You can run this script with the following command:

```bash
python scripts/preparation_script_protein_sequence.py assets/configs/protein_sequence.yaml
```

### Configuration

```yaml
  # Which uniprot dataset to download must be one of the following:
  # "UniprotKB_reviewed", "UniprotKB_unreviewed", "UniRef100", "UniRef90", "UniRef50", "UniParc"
  dataset: "UniRef50"

  # Output directory where the preparation will be made
  output_dir: "/nasa/datasets/riken/projects/fundamental_models_202407/uniprot/"

  # If True use md5 to check if a file needs to be downloaded again, using md5
  # is very time consuming for large file. Otherwise we only check if the path exists.
  use_md5: False

  # Special case for Uniparc download, num of worker to use.
  num_worker: 4

  # Number of sequence per files for raw files and parquet. It also reflex the number
  # of sequence loaded in memory during the processing of those files.
  max_lines_per_file: 10**6
```
### Outputs

The output will be the a subdir of the output_dir containing a dataset name directory (ex: uniprot_50) containing the rest of the file:

- Archive file, for uniprot a `archive` dir will be creating containing all the files
- A fasta file extracted from the archive, for uniprot a `fasta_file` directory will be created containing all the file.
- A `raw_files` directory containing multiple file with one protein sequence per line.
- A `parquet_files` directory, containing two column parquet file tokenized sequence ("token") and the number of ("token_count")
- A `token_counts.pkl` file which contains a list of int corresponding to token_count for computing statistics of the dataset.

### Separate scripts

The processing of Uniprot is separate in 3 separate scripts. These scripts except the result
of precedding directory to be present in the `output_dir` if that's not the case the scripts won't work.

- `src/protein_sequence/dataset/uniprot/uniprot_download.py`

    Will download all uniprot files and extract them to fasta files.

- `src/protein_sequence/dataset/uniprot/fasta_to_raw.py`

    Generate the `raw_files` directory containing smaller raw file of size `max_lines_per_file`

- `src/protein_sequence/dataset/tokenizer.py`

    Convert the raw files in parquet files of tokens in the `parquet_files` directory. The ESM Tokenizer is used.
    The `token_counts.pkl` is also generated.

<!-- ------------------------------------------------------------------------------------------------------------- -->
## RNA

You can call this script with the following command:

```bash
python scripts/preparation_script_rna.py assets/configs/rna.yaml
```

One limitation of the rna sequence task is the fact that the sequence data are continuous and therefore
it creates a challenge for the tokenization and the model. The tokenization is taking into account in the following scripts.
However we could not provide a gpt2 training example, since the gpt2 masking is non trivial and
developing the proper architecture for such a task is past the goal of this project.
See section 3 of the scFormer paper for the details of specific of the model.

### Configuration

```yaml
data_preparation:
  # Output directory where the preparation will be made
  output_dir: "/nasa/datasets/riken/projects/fundamental_models_202407/cellxgene"

  # Special case for Uniparc download, num of worker to use.
  num_worker: 8

  # Size of list of ids to give to each worker, save file will have `size_workload` number of ids in them.
  size_workload: 10000

  # Version of the CellxGene census
  census_version: "2023-12-15"

  # Filter condition to filter genes with few counts across a dataset.
  min_counts_genes: 2
```

### Outputs

This script will download the cellxgene dataset.
There will be multiple directory generate in the output_dir provided in the configuration

- `metadata_preparation`: containing `.tsv` files with ids to download
- `tissue_list.tsv`: List of tissue taken into account.
- `download_dir`: Raw archive file downloaded from the cellxgene database
- `extract`: h5ad file extracted from the archives
- `parquet_files`: parquet files containing tokenized gene and expression values
- `gene_vocab.json`: Dict of gene and their token id

### Separate scripts

The is 4 separate scripts for cellxgene downloading.

- `src/rna/dataset/cellxgene/script/build_list.py`

    Generate `metadata_preparation` and `tissue_list.tsv` to prepare the download.

- `src/rna/dataset/cellxgene/script/download.py`

    Actual downloading of the data in `download_dir` directory.

- `src/rna/dataset/cellxgene/script/conv.py`

    Extract h5ad files form the archived in `download_dir` and save them to the `extract` directory

- `src/rna/dataset/cellxgene/script/tokenization.py`

    Create the gene token vocabulary, filter genes have a counts under `min_counts_genes` and save
    gene tokens and expression value in a tuple in a parquet file. All parquet files are saved
    in `parquet_files` the inputs are the extracted h5ad files.


# Training of GPT2 model

For the GPT2 base code we use the [nanoGPT](https://github.com/karpathy/nanoGPT) code. We only change
the way the dataset is loaded.

In order to train a gpt2 model with one the dataset, you will need to run the `prepare_gpt2.py` script in
`{task}/dataset`, for example for `protein_sequence` you can run the following command.

```bash
python src/protein_sequence/dataset/prepare_gpt2.py assets/configs/protein_sequence.yaml
```

This will prepare the dataset in batch and make sure the context_size, here of 1024
is filled without any padding.

Then the training can be launch, for the prepared protein sequence dataset, by running the following:

```bash
python gpt2/train.py gpt2/data/protein_sequence/train_gpt2_config.py
```

this will train a modeland save it in outputdir.

Then you can sample some example with the following

```bash
python gpt2/sample.py gpt2/data/protein_sequence/train_gpt2_config.py
```
