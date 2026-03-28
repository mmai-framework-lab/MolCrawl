# Compounds Workflow Verification Notes

## Environment Setup

Running the original environment setup failed at `flash_attn` and `biopython` installation.
The environment was created after removing these packages from `environment.yaml`.

```bash
git clone https://github.com/deskull-m/riken-dataset-fundational-model.git
cd riken-dataset-fundational-model
conda env create --name molcrawl --file=environment.yaml

pip install fcd
pip install guacamol
pip install deepchem
```

## GPT-2 Workflow

### Data Preparation

1. Download GuacaMol train/valid/test files and place them in:
   `./benchmark/GuacaMol`

2. Tokenizer update:
   - Updated `CompoundsTokenizer` in `./molcrawl/compounds/utils/tokenizer.py`
   - Added truncation/padding based on `max_len`

3. Max length update:
   - Set `max_len = 128` in `./assets/configs/compounds.yaml`
   - Rationale: GuacaMol molecules have max token length around 102

4. Tokenization:
   - Modified `prepare_gpt2.py` to disable packing
   - Converted GuacaMol into Hugging Face dataset format
   - Output: `./benchmark/GuacaMol/compounds/training_ready_hf_dataset`

```bash
python ./molcrawl/compounds/dataset/prepare_gpt2_.py ./assets/configs/compounds.yaml
```

### Config Changes

Updated files such as `./gpt2/configs/compounds/train_gpt2_config.py`:

- `dataset_dir`: set to preprocessed GuacaMol path
- `batch_size * gradient_accumulation = 256`
- `max_iter`: adjusted for ~10 epochs
- `eval_interval`, `eval_iters`: set per epoch
- `eos_token`: updated to `sep_token=13`
- Added `dataset_params`
- Set `init_from = "fine-tuning"`
- Set `checkpoint_path` to pretrained model

### Fine-tuning (GPT-2)

Training outputs are stored under `./benchmark/GuacaMol`.

```bash
CUDA_VISIBLE_DEVICES=0 python gpt2/train.py gpt2/configs/compounds/train_gpt2_config.py
CUDA_VISIBLE_DEVICES=0 python gpt2/train.py gpt2/configs/compounds/train_gpt2_medium_config.py
CUDA_VISIBLE_DEVICES=0 python gpt2/train.py gpt2/configs/compounds/train_gpt2_large_config.py
CUDA_VISIBLE_DEVICES=0 python gpt2/train.py gpt2/configs/compounds/train_gpt2_xl_config.py
```

### Molecule Generation

1. Updated `GPT.generate` in `./gpt2/model.py` to stop when `eos_token` appears.
2. Computed first-token frequency from `./benchmark/GuacaMol/guacamol_v1_train.smiles` and hardcoded it in `./gpt2/sample_compound.py`.
3. Generated 100,000 compounds to `generated_compounds.txt` under directories such as `./benchmark/GuacaMol/small/`.

```bash
CUDA_VISIBLE_DEVICES=0 python ./gpt2/sample_compound.py ./gpt2/configs/compounds/train_gpt2_config.py
CUDA_VISIBLE_DEVICES=0 python ./gpt2/sample_compound.py ./gpt2/configs/compounds/train_gpt2_medium_config.py
CUDA_VISIBLE_DEVICES=0 python ./gpt2/sample_compound.py ./gpt2/configs/compounds/train_gpt2_large_config.py
CUDA_VISIBLE_DEVICES=0 python ./gpt2/sample_compound.py ./gpt2/configs/compounds/train_gpt2_xl_config.py
```

### GPT-2 Evaluation

Patched library code in `guacamol/utils/chemistry.py`:

```diff
- from scipy import histgram
+ from numpy import histgram
```

Evaluation notebook:

- `./benchmark/GuacaMol/guacamol_evaluation.ipynb`

## BERT Workflow

### Fine-tuning (BERT)

Ran 3 seeds for BERT with and without pretraining.
Evaluated on MoleculeNet.

Output path:
`./benchmark/MoleculeNet/small/{pretrain_or_not}/{benchmark}/{seed}`

```bash
python ./bert/fine-tuning.py
```

### BERT Evaluation

Collected metrics from `test_score.txt` for each run and computed 3-seed averages.
