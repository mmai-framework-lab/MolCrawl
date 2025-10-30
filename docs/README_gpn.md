# Genome sequence training with mixed species GPN-MSA

This folder contains the code for training and inferencing with the Genome sequence with mixed species GPN-MSA for both BERT and GPT2 models.

# Installation

For using these models, you need to create a different conda environment by doing `conda create -n new_environment_name python=3.11`. Once you have creted the enviroment, install the package by going into the folder that contains this readme, and then running `pip install --no-build-isolation -e .`

# Data Preparation

Download the MSA dataset from https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip and uncompress it. The path to the uncompressed folder 89.zarr should be provided in the argument `--msa_path` in the next section. 

# Usage

## PreTraining the Models

We provide some predefined launch commands for the training. A more detailed information of each parameter can be found in the [next section](#parameters-information).
NOTE: set the parameter `--msa_path` to the location of your `89.zarr` folder, downloaded in the [Data Preparation](#data-preparation).

## Train BERT

Run baseline

`python -m gpn.msa.train_bert --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name bert_msa --output_dir checkpoints --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250  --evaluation_strategy steps --eval_steps 250 --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 1 --load_best_model_at_end --overwrite_output_dir --model_type GPNBert --config_overrides n_aux_features=445 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 216 --per_device_eval_batch_size 382 --gradient_accumulation_steps 8 --torch_compile --save_safetensors False`

Alternatively, you can run `python -m gpn.msa.train_bert configs/msa_bert_small.json`.

In order to train for larger sizes change the `--config_overrides` parameter from `n_aux_features=445` to:

* for medium size: `--config_overrides n_aux_features=445,num_hidden_layers=24,embedding_size=1024,num_attention_heads=16 `
* for large size: `--config_overrides n_aux_features=445,num_hidden_layers=36,embedding_size=1280,num_attention_heads=20 `

as:

### Medium Size

`python -m gpn.msa.train_bert --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name bert_msa --output_dir checkpoints --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250  --evaluation_strategy steps --eval_steps 250 --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 1 --load_best_model_at_end --overwrite_output_dir --model_type GPNBert --config_overrides n_aux_features=445,num_hidden_layers=24,embedding_size=1024,num_attention_heads=16 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 216 --per_device_eval_batch_size 382 --gradient_accumulation_steps 8 --torch_compile --save_safetensors False`

Alternatively, you can run `python -m gpn.msa.train_bert configs/msa_bert_mid.json`.

### Large Size

`python -m gpn.msa.train_bert --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name bert_msa --output_dir checkpoints --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250  --evaluation_strategy steps --eval_steps 250 --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 1 --load_best_model_at_end --overwrite_output_dir --model_type GPNBert --config_overrides n_aux_features=445,num_hidden_layers=36,embedding_size=1280,num_attention_heads=20 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 216 --per_device_eval_batch_size 382 --gradient_accumulation_steps 8 --torch_compile --save_safetensors False`

Alternatively, you can run `python -m gpn.msa.train_bert configs/msa_bert_large.json`.


## Train GPT2

Run

`python -m gpn.msa.train_gpt2 --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name gpt_msa --output_dir checkpoints_gpngpt2 --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250 --eval_strategy steps --eval_steps 250 --label_names labels --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 2 --load_best_model_at_end --overwrite_output_dir --model_type GPNGPT2 --config_overrides n_aux_features=445 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 8 --save_safetensors False --torch_compile`

Alternatively, you can run `python -m gpn.msa.train_gpt2 configs/msa_gpt2_small.json`.

In order to train for larger sizes change the `--config_overrides` parameter from `n_aux_features=445` to:

* for medium size: `--config_overrides n_aux_features=445,num_hidden_layers=24,hidden_size=1024,num_attention_heads=16 `
* for large size: `--config_overrides n_aux_features=445,num_hidden_layers=36,hidden_size=1280,num_attention_heads=20 `


as:

### Medium Size

`python -m gpn.msa.train_gpt2 --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name gpt_msa --output_dir checkpoints_gpngpt2 --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250 --eval_strategy steps --eval_steps 250 --label_names labels --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 2 --load_best_model_at_end --overwrite_output_dir --model_type GPNGPT2 --config_overrides n_aux_features=445,num_hidden_layers=24,hidden_size=1024,num_attention_heads=16 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 8 --save_safetensors False --torch_compile`

Alternatively, you can run `python -m gpn.msa.train_gpt2 configs/msa_gpt2_mid.json`.

### Large Size

`python -m gpn.msa.train_gpt2 --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name gpt_msa --output_dir checkpoints_gpngpt2 --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250 --eval_strategy steps --eval_steps 250 --label_names labels --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 2 --load_best_model_at_end --overwrite_output_dir --model_type GPNGPT2 --config_overrides n_aux_features=445,num_hidden_layers=36,hidden_size=1280,num_attention_heads=20 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 8 --save_safetensors False --torch_compile`

Alternatively, you can run `python -m gpn.msa.train_gpt2 configs/msa_gpt2_large.json`.

# Parameters' Information


```bash
python -m gpn.msa.train_gpt2 --do_train --do_eval --report_to tensorboard --fp16 ...
```

### 🚀 Core Training Options

* **`--do_train`**
  Enables training mode.

* **`--do_eval`**
  Enables evaluation during training.

* **`--report_to tensorboard`**
  Logs training metrics to TensorBoard.

* **`--fp16`**
  Enables mixed-precision training (16-bit floats), which can reduce memory usage and improve performance on supported GPUs.

* **`--prediction_loss_only true`**
  Only compute and return the loss, not the full model outputs (saves memory).


### 📚 Dataset & Paths

* **`--dataset_name songlab/gpn-msa-sapiens-dataset`**
  Hugging Face dataset to load for training and evaluation.

* **`--msa_path gpn/analysis/89.zarr`**
  Path to the input MSA data in Zarr format.


### 🧪 Experiment Management

* **`--run_name gpt_msa`**
  Name of the run, used in logs and experiment tracking.

* **`--output_dir checkpoints_gpngpt2`**
  Directory where model checkpoints and logs will be saved.

* **`--overwrite_output_dir`**
  Overwrites the contents of the output directory if it exists.

* **`--load_best_model_at_end`**
  Loads the best checkpoint (based on evaluation metrics) at the end of training.


### 🧠 Model & Loss Configuration

* **`--model_type GPNGPT2`**
  Specifies the model class to use.

* **`--config_overrides n_aux_features=445`**
  Overrides model config to set the number of auxiliary input features.

* **`--use_aux_features True`**
  Enables usage of auxiliary features during training.

* **`--soft_masked_loss_weight_train 0.1`**
  Weight for soft-masked loss during training. Typically used in masked modeling for continuous inputs.

* **`--soft_masked_loss_weight_evaluation 0.1`**
  Weight for soft-masked loss during evaluation.

* **`--weight_conserved True`**
  Apply additional loss weighting to conserved positions in the MSA.

* **`--flip_nonconserved True`**
  Introduce random flipping of non-conserved tokens for data augmentation.

* **`--remove_unused_columns False`**
  Keep all columns from the dataset, not just those used by the model (useful when custom fields are required).


### ⚙️ Optimization & Scheduling

* **`--weight_decay 0.01`**
  L2 weight regularization to prevent overfitting.

* **`--optim adamw_torch`**
  Optimizer to use (AdamW implemented in PyTorch).

* **`--learning_rate 1e-4`**
  Initial learning rate for the optimizer.

* **`--lr_scheduler_type cosine`**
  Scheduler type to decay learning rate with a cosine annealing schedule.

* **`--warmup_steps 1000`**
  Number of warmup steps where the learning rate increases linearly.

* **`--max_steps 30000`**
  Total number of training steps (batches).


### 🧵 Data Loading & Parallelism

* **`--dataloader_num_workers 8`**
  Number of subprocesses for data loading.


### 💾 Checkpointing & Evaluation

* **`--save_strategy steps`**
  Save checkpoints every fixed number of steps.

* **`--save_steps 250`**
  Interval (in steps) to save model checkpoints.

* **`--save_total_limit 2`**
  Keep only the 2 most recent checkpoints to save disk space.

* **`--eval_strategy steps`**
  Evaluate the model at regular step intervals.

* **`--eval_steps 250`**
  Interval (in steps) to run evaluation.

* **`--logging_steps 10`**
  Log training metrics every 10 steps.


### 🏷️ Labels & Features

* **`--label_names labels`**
  Specifies which column(s) in the dataset to use as the label during training.


### 🧮 Batch Size & Accumulation

* **`--per_device_train_batch_size 128`**
  Training batch size per device (GPU or CPU).

* **`--per_device_eval_batch_size 256`**
  Evaluation batch size per device.

* **`--gradient_accumulation_steps 8`**
  Accumulate gradients over 8 steps before updating model weights.
  This enables effective large-batch training even with memory constraints.


### ⚡ Performance & Precision

* **`--save_safetensors False`**
  Disables saving checkpoints in `.safetensors` format (saves in traditional PyTorch `.bin` format).

* **`--torch_compile`**
  Uses `torch.compile()` to optimize model execution with PyTorch 2.0 (experimental but can offer performance boosts).


## (Extra) Running Inference with the Models

To see the models working, an inference logic was added as an extra. Once the models are trained, refer to `gpn/examples`.