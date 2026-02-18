# GPT2 Test

This folder is for validating the training ability of GPT2 on subsets of the prepared datasets.
The model trains on a small subset of the data. By overfitting on the training set, we can validate if the model is able to learn from the dataset.

## Usage

1. Prepare your dataset subset by running `python gpt2/configs/<dataset>/prepare.py path/to/the/tokenized/dataset`

This will load the dataset, sample a subset, and create batches of the same length.
Note: the parameters `--training-set-subset-len` and `--test-set-subset-len` can be used to select the subset size. If < 1 taken as fracation of full data. If > 1 taken as number of samples.

1. Train the model by running `python gpt2/train.py path/to/corresponding/dataset/train_gpt2_config.py`

Inside each `data/<dataset>` folder, there is a file named `train_gpt2_config.py`, which contains parameters to train GPT2 in that dataset. For example: `python gpt2/train.py riken-dataset-fundational-model/gpt2/configs/molecule_nl/train_gpt2_large_config.py` will train the large GPT2 model on the molecule_nl dataset.

Running this will lunch a training job, and output results in the path `out/ckpt.pt

NOTE: If you have `torchrun`, you can run the model over multiple GPUs like:

`torchrun --standalone --nproc_per_node=4 config_file.py`

to run over 4 GPUs.

To run with DDP on 4 gpus across 2 nodes, example:

- Run on the first (master) node with example IP 123.456.123.456:
  `torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 config_file.py`
- Run on the worker node:
  `torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 config_file.py`
  (If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

1. Generate a sample from the trained checkpoint running `python gpt2/sample.py {config.py}`. This should be the same config file that you used for trainig, for example `python gpt2/sample.py riken-dataset-fundational-model/gpt2/configs/molecule_nl/train_gpt2_large_config.py` for the exmaple in step 2.

## 🛠️ Configuration Parameters

In each `path/to/corresponding/dataset/train_gpt2_config.py` config, you can set the following parameters.

### IMPORTANT SETTING PARAMETERS

#### tokenizer_path

Some datasets have this parameter in the config.py. If so, they require you to set this parameter accordingly to the tokenizer which resulted from the data-preprocessing.

The instructions to set it are as follows:

- For Genome Sequence: select the `spm_tokenizer.model` file generated after running the script `scripts/preparation_script_genome_sequence.py`. It will be in the folder you defined in the `assets/configs/genome_sequence.yaml` under `output_dir`.
- For Compounds: select the `vocab.txt` file generated after running the script `scripts/preparation_script_compounds.py`. It will be in the path you defined in the `assets/configs/compounds.yaml` under `vocab_path`.
- For Molecule Natural Language: It does not require you to set it up.
- For Protein Sequence: It does not require you to set it up.

#### dataset_dir

The path to your processed dataset. This is the output of [Usage](#usage), step 1. When you run this file, it will print the output of the dataset, which you should match to this parameter.

### 📊 Logging

- **`tensorboard`**
  Enables logging of training metrics (e.g., loss, learning rate) to TensorBoard.

- **`tensorboard_dir`**
  Directory where TensorBoard logs will be saved.

### Output Weights

- **`out_dir`**
  Directory where training outputs (e.g., checkpoints, logs) will be saved.

### 📦 Batch Settings

- **`batch_size`**
  Number of samples per batch per GPU during training.

- **`block_size`**
  Maximum sequence length (in tokens) the model processes at once.

- **`gradient_accumulation_steps`**
  Number of steps to accumulate gradients before performing a backward pass and optimizer step.
  Allows for an effective large batch size without exceeding GPU memory.

> 🧮 **Effective Batch Size:**
> `batch_size × block_size × gradient_accumulation_steps × num_GPUs`
> Example: `8 × 1024 × 80 × 8 = 524,288 tokens`

### 📉 Learning Rate Schedule

- **`max_iters`**
  Total number of training iterations (i.e., batches).

- **`lr_decay_iters`**
  Number of iterations over which the learning rate linearly decays from `learning_rate` to `min_lr`.

- **`warmup_iters`**
  Number of iterations during which the learning rate warms up from 0 to `learning_rate`.

- **`learning_rate`**
  Peak learning rate reached after the warmup period.

- **`min_lr`**
  Minimum learning rate to decay to by the end of training.

### 🧪 Evaluation & Logging

- **`eval_interval`**
  Frequency (in iterations) of evaluation during training.

- **`eval_iters`**
  Number of evaluation batches used to compute validation metrics.

- **`log_interval`**
  Frequency (in iterations) at which training metrics are printed or logged.

### 🧹 Regularization

- **`weight_decay`**
  L2 weight regularization strength to prevent overfitting.
