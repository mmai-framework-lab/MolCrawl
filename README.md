# Installation
1. Install the package: `pip install --no-build-isolation -e .`

# Usage

Use each script depending on the dataset you want to process for an LLM.

## Molecules

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
