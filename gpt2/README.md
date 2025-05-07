# GPT2 Test

This folder is for validating the training ability of GPT2 on subsets of the prepared datasets.
The model trains on a small subset of the data. By overfitting on the training set, we can validate if the model is able to learn from the dataset.

## Usage

1. Prepare your dataset subset by running `python gpt2/data/<dataset>/prepare.py path/to/the/tokenized/dataset`

This will load the dataset, sample a subset, and create batches of the same length.
Note: the parameters `--training-set-subset-len` and `--test-set-subset-len` can be used to select the subset size. If < 1 taken as fracation of full data. If > 1 taken as number of samples.

2. Train the model by running `python gpt2/train.py path/to/corresponding/dataset/train_gpt2_config.py`

Inside each `data/<dataset>` folder, there is a file named `train_gpt2_config.py`, which contains parameters to train GPT2 in that dataset. For example: `python gpt2/train.py riken-dataset-fundational-model/gpt2/data/molecule_nl/train_gpt2_large_config.py` will train the large GPT2 model on the molecule_nl dataset.

Running this will lunch a training job, and output results in the path `out/ckpt.pt

3. Generate a sample from the trained checkpoint running `python gpt2/sample.py {config.py}`. This should be the same config file that you used for trainig, for example `python gpt2/sample.py riken-dataset-fundational-model/gpt2/data/molecule_nl/train_gpt2_large_config.py` for the exmaple in step 2.
