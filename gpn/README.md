# Genome sequence training with mixed species GPN-MSA

This folder contains the code for training and inferencing with the Genome sequence with mixed species GPN-MSA for both BERT and GPT2 models.

# Installation

For using these models, you need to create a different conda environment. Once you have creted the enviroment, install the package by going into the folder that contains this readme, and then running `pip install --no-build-isolation -e .`


# Usage

# Running Inference with the Models

Running inference with the pretrained models can be done by following the examples in `gpn/examples`.

## PreTraining the Models

### Train BERT

Run baseline

`python -m gpn.msa.train --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name bert_msa --output_dir checkpoints --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250  --evaluation_strategy steps --eval_steps 250 --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 1 --load_best_model_at_end --overwrite_output_dir --model_type GPNBert --config_overrides n_aux_features=445 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 216 --per_device_eval_batch_size 382 --gradient_accumulation_steps 8 --torch_compile --save_safetensors False`

In order to train for larger sizes change the `--config_overrides` parameter from `n_aux_features=445` to:

* for medium size: `--config_overrides n_aux_features=445,num_hidden_layers=24,embedding_size=1024,num_attention_heads=16 `
* for large size: `--config_overrides n_aux_features=445,num_hidden_layers=36,embedding_size=1280,num_attention_heads=20 `

### Train GPT2

Run

`python -m gpn.msa.train_gpt2 --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name gpt_msa --output_dir checkpoints_gpngpt2 --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250 --eval_strategy steps --eval_steps 250 --label_names labels --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 2 --load_best_model_at_end --overwrite_output_dir --model_type GPNGPT2 --config_overrides n_aux_features=445 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 8 --save_safetensors False --torch_compile`

In order to train for larger sizes change the `--config_overrides` parameter from `n_aux_features=445` to:

* for medium size: `--config_overrides n_aux_features=445,num_hidden_layers=24,hidden_size=1024,num_attention_heads=16 `
* for large size: `--config_overrides n_aux_features=445,num_hidden_layers=36,hidden_size=1280,num_attention_heads=20 `
