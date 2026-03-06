# 混合種 GPN-MSA を用いた Genome Sequence 学習

このフォルダには、混合種 GPN-MSA を使って、BERT および GPT-2 モデルの学習と推論を行うためのコードが含まれています。

## インストール

これらのモデルを使うには、別の conda 環境を作成する必要があります。`conda create -n new_environment_name python=3.11` を実行してください。
環境作成後、この README があるフォルダへ移動し、`pip install --no-build-isolation -e .` を実行してパッケージをインストールします。

## データ準備

MSA データセットを <https://Huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip> からダウンロードして解凍してください。
解凍後の `89.zarr` フォルダへのパスを、次節の `--msa_path` 引数に指定します。

## 使い方

### モデルの事前学習

学習用の定義済み起動コマンドを用意しています。各パラメータの詳細は [次のセクション](#パラメータ情報) を参照してください。
注意: [データ準備](#データ準備) で用意した `89.zarr` の場所を `--msa_path` に設定してください。

### BERT を学習

ベースライン実行:

`python -m gpn.msa.train_bert --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name bert_msa --output_dir checkpoints --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250  --evaluation_strategy steps --eval_steps 250 --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 1 --load_best_model_at_end --overwrite_output_dir --model_type GPNBert --config_overrides n_aux_features=445 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 216 --per_device_eval_batch_size 382 --gradient_accumulation_steps 8 --torch_compile --save_safetensors False`

または `python -m gpn.msa.train_bert configs/msa_bert_small.json` を実行できます。

より大きいサイズを学習する場合、`--config_overrides` を `n_aux_features=445` から以下へ変更します。

- medium: `--config_overrides n_aux_features=445,num_hidden_layers=24,embedding_size=1024,num_attention_heads=16`
- large: `--config_overrides n_aux_features=445,num_hidden_layers=36,embedding_size=1280,num_attention_heads=20`

以下のように実行します。

#### Medium Size (BERT)

`python -m gpn.msa.train_bert --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name bert_msa --output_dir checkpoints --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250  --evaluation_strategy steps --eval_steps 250 --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 1 --load_best_model_at_end --overwrite_output_dir --model_type GPNBert --config_overrides n_aux_features=445,num_hidden_layers=24,embedding_size=1024,num_attention_heads=16 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 216 --per_device_eval_batch_size 382 --gradient_accumulation_steps 8 --torch_compile --save_safetensors False`

または `python -m gpn.msa.train_bert configs/msa_bert_mid.json` を実行できます。

#### Large Size (BERT)

`python -m gpn.msa.train_bert --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name bert_msa --output_dir checkpoints --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250  --evaluation_strategy steps --eval_steps 250 --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 1 --load_best_model_at_end --overwrite_output_dir --model_type GPNBert --config_overrides n_aux_features=445,num_hidden_layers=36,embedding_size=1280,num_attention_heads=20 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 216 --per_device_eval_batch_size 382 --gradient_accumulation_steps 8 --torch_compile --save_safetensors False`

または `python -m gpn.msa.train_bert configs/msa_bert_large.json` を実行できます。

### GPT-2 を学習

実行:

`python -m gpn.msa.train_gpt2 --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name gpt_msa --output_dir checkpoints_gpngpt2 --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250 --eval_strategy steps --eval_steps 250 --label_names labels --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 2 --load_best_model_at_end --overwrite_output_dir --model_type GPNGPT2 --config_overrides n_aux_features=445 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 8 --save_safetensors False --torch_compile`

または `python -m gpn.msa.train_gpt2 configs/msa_gpt2_small.json` を実行できます。

より大きいサイズを学習する場合、`--config_overrides` を `n_aux_features=445` から以下へ変更します。

- medium: `--config_overrides n_aux_features=445,num_hidden_layers=24,hidden_size=1024,num_attention_heads=16`
- large: `--config_overrides n_aux_features=445,num_hidden_layers=36,hidden_size=1280,num_attention_heads=20`

以下のように実行します。

#### Medium Size (GPT-2)

`python -m gpn.msa.train_gpt2 --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name gpt_msa --output_dir checkpoints_gpngpt2 --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250 --eval_strategy steps --eval_steps 250 --label_names labels --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 2 --load_best_model_at_end --overwrite_output_dir --model_type GPNGPT2 --config_overrides n_aux_features=445,num_hidden_layers=24,hidden_size=1024,num_attention_heads=16 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 8 --save_safetensors False --torch_compile`

または `python -m gpn.msa.train_gpt2 configs/msa_gpt2_mid.json` を実行できます。

#### Large Size (GPT-2)

`python -m gpn.msa.train_gpt2 --do_train --do_eval --report_to tensorboard --fp16 --prediction_loss_only true --dataset_name songlab/gpn-msa-sapiens-dataset --msa_path gpn/analysis/89.zarr --run_name gpt_msa --output_dir checkpoints_gpngpt2 --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.1 --weight_decay 0.01 --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine --seed 42 --dataloader_num_workers 8 --save_strategy steps --save_steps 250 --eval_strategy steps --eval_steps 250 --label_names labels --logging_steps 10 --max_steps 30_000 --warmup_steps 1000 --save_total_limit 2 --load_best_model_at_end --overwrite_output_dir --model_type GPNGPT2 --config_overrides n_aux_features=445,num_hidden_layers=36,hidden_size=1280,num_attention_heads=20 --use_aux_features True --weight_conserved True --flip_nonconserved True --remove_unused_columns False --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 8 --save_safetensors False --torch_compile`

または `python -m gpn.msa.train_gpt2 configs/msa_gpt2_large.json` を実行できます。

## パラメータ情報

```bash
python -m gpn.msa.train_gpt2 --do_train --do_eval --report_to tensorboard --fp16 ...
```

### コア学習オプション

- **`--do_train`**
  学習モードを有効化します。

- **`--do_eval`**
  学習中の評価を有効化します。

- **`--report_to tensorboard`**
  学習メトリクスを TensorBoard に出力します。

- **`--fp16`**
  混合精度学習（16-bit）を有効化します。対応 GPU ではメモリ削減と高速化が期待できます。

- **`--prediction_loss_only true`**
  モデル出力全体ではなく loss のみを計算・返却します（メモリ節約）。

### データセット・パス

- **`--dataset_name songlab/gpn-msa-sapiens-dataset`**
  学習・評価に使用する Hugging Face データセット。

- **`--msa_path gpn/analysis/89.zarr`**
  Zarr 形式 MSA データへのパス。

### 実験管理

- **`--run_name gpt_msa`**
  ログや実験トラッキングで使われる実行名。

- **`--output_dir checkpoints_gpngpt2`**
  チェックポイントとログの保存先。

- **`--overwrite_output_dir`**
  既存の出力ディレクトリ内容を上書きします。

- **`--load_best_model_at_end`**
  学習終了時に評価指標ベースで最良チェックポイントを読み込みます。

### モデル・損失設定

- **`--model_type GPNGPT2`**
  使用するモデルクラスを指定します。

- **`--config_overrides n_aux_features=445`**
  補助入力特徴量数などのモデル設定を上書きします。

- **`--use_aux_features True`**
  学習時に補助特徴量を利用します。

- **`--soft_masked_loss_weight_train 0.1`**
  学習時の soft-masked loss の重み。

- **`--soft_masked_loss_weight_evaluation 0.1`**
  評価時の soft-masked loss の重み。

- **`--weight_conserved True`**
  MSA の保存領域（conserved positions）に追加重みを付けます。

- **`--flip_nonconserved True`**
  非保存トークンをランダムに反転し、データ拡張を行います。

- **`--remove_unused_columns False`**
  モデル未使用カラムも保持します（カスタム項目が必要な場合に有用）。

### 最適化・スケジューリング

- **`--weight_decay 0.01`**
  過学習抑制のための L2 正則化。

- **`--optim adamw_torch`**
  PyTorch 実装の AdamW オプティマイザー。

- **`--learning_rate 1e-4`**
  初期学習率。

- **`--lr_scheduler_type cosine`**
  cosine annealing による学習率減衰。

- **`--warmup_steps 1000`**
  学習率を線形に増加させるウォームアップステップ数。

- **`--max_steps 30000`**
  学習総ステップ数。

### データ読み込み・並列化

- **`--dataloader_num_workers 8`**
  データ読み込みに使うサブプロセス数。

### チェックポイント・評価

- **`--save_strategy steps`**
  固定ステップ間隔でチェックポイントを保存。

- **`--save_steps 250`**
  チェックポイント保存間隔（ステップ）。

- **`--save_total_limit 2`**
  直近2つのみ保持し、ディスク使用量を抑えます。

- **`--eval_strategy steps`**
  固定ステップ間隔で評価を実行。

- **`--eval_steps 250`**
  評価間隔（ステップ）。

- **`--logging_steps 10`**
  ログ出力間隔（10ステップごと）。

### ラベル・特徴量

- **`--label_names labels`**
  学習時にラベルとして扱うデータセット列名を指定。

### バッチサイズ・勾配累積

- **`--per_device_train_batch_size 128`**
  デバイスごとの学習バッチサイズ。

- **`--per_device_eval_batch_size 256`**
  デバイスごとの評価バッチサイズ。

- **`--gradient_accumulation_steps 8`**
  重み更新前に 8 ステップ分勾配を蓄積します。

### 性能・精度

- **`--save_safetensors False`**
  `.safetensors` 形式での保存を無効化し、従来の PyTorch `.bin` 形式で保存します。

- **`--torch_compile`**
  PyTorch 2.0 の `torch.compile()` による最適化を使用します（実験的）。

## 付録: 推論実行

モデル動作確認のため、追加で推論ロジックも用意されています。
学習後は `gpn/examples` を参照してください。
