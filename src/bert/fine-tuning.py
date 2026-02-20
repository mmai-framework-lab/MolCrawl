import copy
from functools import partial

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, DatasetDict
from deepchem import molnet
from deepchem.trans import NormalizationTransformer
from rdkit import Chem
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer

# 共通環境チェックモジュールを追加
from src.utils.environment_check import check_learning_source_dir

# -----------------------------
# 既存の前処理ヘルパ
# -----------------------------
params = Chem.SmilesParserParams()
params.removeHs = True


def canonical(mol):
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    mol = Chem.MolFromSmiles(smiles, params=params)
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def df_to_ds(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def metric_acc(predictions, references):
    return {"accuracy": (np.array(predictions) == np.array(references)).mean()}


def metric_mae(predictions, references):
    return {"mae": np.mean(np.abs(np.array(predictions) - np.array(references)))}


def metric_mse(predictions, references):
    return {"mse": np.mean((np.array(predictions) - np.array(references)) ** 2)}


# -----------------------------
# メトリクス
# -----------------------------


def compute_metrics(eval_pred, std=None):
    logits, labels = eval_pred
    labels = np.array(labels)

    if task_type == "regression":
        preds = np.squeeze(logits)
        labels_f = labels.astype(np.float64)
        mse = metric_mse(predictions=preds, references=labels_f)["mse"]
        mae = metric_mae(predictions=preds, references=labels_f)["mae"]
        rmse = float(mse**0.5)
        # if std is not None, scale to original
        if std is not None:
            mse *= std**2
            rmse *= std
            mae *= std
        del logits, labels, preds, labels_f
        torch.cuda.empty_cache()
        return {"mse": mse, "rmse": rmse, "mae": mae}

    elif task_type == "classification":
        # CE: logits shape [B, 2]
        pred_ids = np.argmax(logits, axis=-1)
        acc = metric_acc(predictions=pred_ids, references=labels)["accuracy"]
        # AUROC（二値）
        try:
            probs_pos = torch.tensor(logits).softmax(dim=-1)[:, 1].numpy()
            auroc = roc_auc_score(labels, probs_pos)
        except Exception:
            auroc = float("nan")
        f1 = f1_score(labels, pred_ids)
        del logits, labels, pred_ids, probs_pos
        torch.cuda.empty_cache()
        return {"accuracy": acc, "f1": f1, "auroc": auroc}

    else:
        # multi-label: logits [B, K], sigmoid -> probs, 0.5 で閾値
        probs = torch.tensor(logits).sigmoid().numpy()
        pred_mt = (probs >= 0.5).astype(int)  # [B, K]
        labels_mt = labels.astype(int)

        # micro/macro F1
        f1_micro = f1_score(labels_mt, pred_mt, average="micro", zero_division=0)
        f1_macro = f1_score(labels_mt, pred_mt, average="macro", zero_division=0)

        # 平均 AUROC（列ごとに算出し、NaN は無視）
        aucs = []
        for k in range(probs.shape[1]):
            y_true = labels_mt[:, k]
            y_prob = probs[:, k]
            # 片方しか出ていない列はスキップ
            if len(np.unique(y_true)) < 2:
                continue
            try:
                aucs.append(roc_auc_score(y_true, y_prob))
            except Exception:
                pass
        mean_auroc = float(np.mean(aucs)) if len(aucs) else float("nan")
        del logits, labels, probs, pred_mt, labels_mt, aucs
        torch.cuda.empty_cache()
        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "mean_auroc": mean_auroc,
        }


def preprocess_data(data, tasks):
    train, valid, test = data
    train_df = pd.DataFrame({"text": [canonical(mol) for mol in train.X]})
    valid_df = pd.DataFrame({"text": [canonical(mol) for mol in valid.X]})
    test_df = pd.DataFrame({"text": [canonical(mol) for mol in test.X]})
    if len(tasks) == 1:
        train_df["label"] = train.y
        valid_df["label"] = valid.y
        test_df["label"] = test.y
    else:
        for i, task in enumerate(tasks):
            train_df[task] = train.y[:, i]
            valid_df[task] = valid.y[:, i]
            test_df[task] = test.y[:, i]
    train_df = train_df.dropna().reset_index(drop=True)
    valid_df = valid_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    return train_df, valid_df, test_df


if __name__ == "__main__":
    # -----------------------------
    # データセット選択（例: tox21 は multi-task）
    # -----------------------------

    for pretrained in [True, False]:
        for seed in range(3):
            for benchmark_name in [
                "bace",
                "bbbp",
                "clearance",
                "clintox",
                "delaney",
                "freesolv",
                "hiv",
                "lipo",
                "qm8",
                "qm9",
                "sider",
                "tox21",
            ]:
                if benchmark_name == "bace":
                    tasks, datasets, transformers = molnet.load_bace_classification(featurizer="raw", transforms=[])
                    task_type = "classification"  # 単一ラベル（0/1）
                elif benchmark_name == "bbbp":
                    tasks, datasets, transformers = molnet.load_bbbp(featurizer="raw")
                    task_type = "classification"
                elif benchmark_name == "clearance":
                    tasks, datasets, transformers = molnet.load_clearance(featurizer="raw", transformers=[])
                    task_type = "regression"
                elif benchmark_name == "clintox":
                    tasks, datasets, transformers = molnet.load_clintox(featurizer="raw")
                    task_type = "multitask_classification"  # 複数ラベル
                elif benchmark_name == "delaney":
                    tasks, datasets, transformers = molnet.load_delaney(featurizer="raw", transformers=[])
                    task_type = "regression"
                elif benchmark_name == "freesolv":
                    tasks, datasets, transformers = molnet.load_freesolv(featurizer="raw", transformers=[])
                    task_type = "regression"
                elif benchmark_name == "hiv":
                    tasks, datasets, transformers = molnet.load_hiv(featurizer="raw")
                    task_type = "classification"
                elif benchmark_name == "lipo":
                    tasks, datasets, transformers = molnet.load_lipo(featurizer="raw", transformers=[])
                    task_type = "regression"
                elif benchmark_name == "qm8":
                    tasks, datasets, transformers = molnet.load_qm8(featurizer="raw", transformers=[])
                    task_type = "regression"
                elif benchmark_name == "qm9":
                    tasks, datasets, transformers = molnet.load_qm9(featurizer="raw", transformers=[])
                    task_type = "regression"
                elif benchmark_name == "sider":
                    tasks, datasets, transformers = molnet.load_sider(featurizer="raw")
                    task_type = "multitask_classification"
                elif benchmark_name == "tox21":
                    tasks, datasets, transformers = molnet.load_tox21(featurizer="raw")
                    task_type = "multitask_classification"

                if task_type == "regression":
                    train, valid, test = datasets
                    transformers = NormalizationTransformer(transform_y=True, dataset=train)  # 統計はtrainから推定
                    train = transformers.transform(train)
                    valid = transformers.transform(valid)
                    test = transformers.transform(test)
                    datasets = (train, valid, test)

                train_df, valid_df, test_df = preprocess_data(datasets, tasks)
                all_df = pd.concat([train_df, valid_df, test_df]).reset_index(drop=True)

                # -----------------------------
                # 設定
                # -----------------------------
                learning_source_dir = check_learning_source_dir()
                output_dir = (
                    f"{learning_source_dir}/compounds/benchmark/MoleculeNet/small/pretrained_{pretrained}/{benchmark_name}"
                )
                learning_rate = 3e-5
                weight_decay = 0.0
                warmup_steps = 0
                epochs = 500
                world_size = 2
                gradient_accumulation_steps = 2 if benchmark_name == "hiv" else 1
                if len(train_df) <= 1000:
                    batch_size = 32 / world_size / gradient_accumulation_steps
                elif len(train_df) <= 5000:
                    batch_size = 256 / world_size / gradient_accumulation_steps
                else:
                    batch_size = 512 / world_size / gradient_accumulation_steps
                batch_size = int(batch_size)

                per_device_eval_batch_size = 512
                if benchmark_name == "hiv":
                    per_device_eval_batch_size = 128
                tokenizer = Tokenizer("assets/molecules/vocab.txt", 512)

                # 最大長の自動見積り（安全に 512 にクリップ）
                all_df["token_length"] = all_df["text"].apply(lambda x, tokenizer=tokenizer: len(tokenizer.encode(x)))
                max_length = int(min(all_df["token_length"].max(), 512))

                # 語彙サイズ（8の倍数へ丸め）
                meta_vocab_size = (len(tokenizer) // 8 + 1) * 8

                # -----------------------------
                # モデル設定：タスク別に num_labels/problem_type を切替
                # -----------------------------
                if task_type == "regression":
                    num_labels = 1
                    problem_type = "regression"
                elif task_type == "classification":
                    # 単一バイナリ分類は CrossEntropy 用に num_labels=2 が扱いやすい（labels は int 0/1）
                    num_labels = 2
                    problem_type = "single_label_classification"
                elif task_type == "multitask_classification":
                    num_labels = len(tasks)  # たとえば tox21 は 12
                    problem_type = "multi_label_classification"
                else:
                    raise ValueError(f"unknown task_type: {task_type}")

                model_config = BertConfig(
                    vocab_size=meta_vocab_size,
                    max_position_embeddings=1024,
                    num_labels=num_labels,
                )

                if pretrained:
                    # Use relative path for pretrained model checkpoint
                    pretrained_model_path = "runs_train_bert_compounds/checkpoint-4000"
                    model = BertForSequenceClassification.from_pretrained(
                        pretrained_model_path,
                        config=model_config,
                    )
                else:
                    model = BertForSequenceClassification(config=model_config)
                model.config.problem_type = problem_type

                # -----------------------------
                # HF Datasets に載せるための前処理
                # - 入力: train_df/valid_df/test_df （text + label列）
                # - 出力: tokenized DatasetDict（labels を正しい型/形に）
                # -----------------------------

                raw_ds = DatasetDict(
                    {
                        "train": df_to_ds(train_df),
                        "validation": df_to_ds(valid_df),
                        "test": df_to_ds(test_df),
                    }
                )

                def preprocess_function(
                    examples,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    task_type=task_type,
                    tasks=tasks,
                ):
                    enc = tokenizer(
                        examples["text"],
                        truncation=True,
                        max_length=max_length,
                    )
                    # ラベルの成形
                    if task_type == "regression":
                        # float スカラー
                        # もともと 'label' カラムに入っている想定
                        labels = [float(x) for x in examples["label"]]
                    elif task_type == "classification":
                        # int スカラー（0/1）
                        labels = [int(x) for x in examples["label"]]
                    else:  # multi-label
                        # 各タスク列を float へ（NaN は 0.0 に置換）
                        labels = []
                        n = len(examples["text"])
                        for i in range(n):
                            row = []
                            for t in tasks:
                                v = examples[t][i]
                                if v is None or (isinstance(v, float) and np.isnan(v)):
                                    v = 0.0
                                row.append(float(v))
                            labels.append(row)
                    enc["labels"] = labels
                    return enc

                # -----------------------------
                # Data collator
                # -----------------------------
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

                # -----------------------------
                # 学習設定（epoch ベース）
                # -----------------------------
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    overwrite_output_dir=True,
                    num_train_epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    warmup_steps=warmup_steps,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    logging_strategy="epoch",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model=(
                        "rmse" if task_type == "regression" else ("auroc" if task_type == "classification" else "mean_auroc")
                    ),
                    greater_is_better=(False if task_type == "regression" else True),
                    report_to="wandb",
                    run_name=f"{benchmark_name}_pretrained{pretrained}_seed{seed}",
                    seed=seed,
                    save_total_limit=2,
                    fp16=True,
                )

                # -----------------------------
                # Trainer
                # -----------------------------
                if benchmark_name in ["qm8", "qm9"]:
                    for i in range(len(tasks)):
                        wandb.init(
                            project="molecule-benchmark",  # プロジェクト名（任意）
                            name=f"{tasks[i]}_pretrained{pretrained}_seed{seed}",  # run名
                            config={
                                "benchmark": benchmark_name,
                                "pretrained": pretrained,
                                "seed": seed,
                            },
                        )
                        training_args.run_name = f"{tasks[i]}_pretrained{pretrained}_seed{seed}"
                        if pretrained:
                            model = BertForSequenceClassification.from_pretrained(
                                pretrained_model_path,
                                config=model_config,
                            )
                        else:
                            model = BertForSequenceClassification(config=model_config)
                        model.config.problem_type = problem_type

                        # rename task col to 'label' for Trainer
                        raw_ds_ = copy.deepcopy(raw_ds)
                        raw_ds_["train"] = raw_ds_["train"].rename_column(tasks[i], "label")
                        raw_ds_["validation"] = raw_ds_["validation"].rename_column(tasks[i], "label")
                        raw_ds_["test"] = raw_ds_["test"].rename_column(tasks[i], "label")
                        print(f"Training for task {tasks[i]} ({i + 1}/{len(tasks)})")
                        training_args.output_dir = f"{output_dir}/task_{tasks[i]}/seed{seed}"

                        tokenized = raw_ds_.map(
                            preprocess_function,
                            batched=True,
                            remove_columns=raw_ds_["train"].column_names,
                        )

                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=tokenized["train"],
                            eval_dataset=tokenized["validation"],
                            tokenizer=None,
                            compute_metrics=partial(compute_metrics, std=transformers.y_stds[i]),
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
                        )

                        trainer.train()
                        trainer.save_model(f"{training_args.output_dir}/best_model")

                        test_score = trainer.evaluate(tokenized["test"])
                        print(f"Test score: {test_score}")
                        with open(f"{training_args.output_dir}/test_score.txt", "w") as f:
                            f.write(str(test_score))
                        wandb.log({"test_score": test_score})
                        wandb.finish()
                else:
                    wandb.init(
                        project="molecule-benchmark",  # プロジェクト名（任意）
                        name=f"{benchmark_name}_pretrained{pretrained}_seed{seed}",  # run名
                        config={
                            "benchmark": benchmark_name,
                            "pretrained": pretrained,
                            "seed": seed,
                        },
                    )
                    tokenized = raw_ds.map(
                        preprocess_function,
                        batched=True,
                        remove_columns=raw_ds["train"].column_names,
                    )
                    training_args.output_dir = f"{output_dir}/seed{seed}"

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=tokenized["train"],
                        eval_dataset=tokenized["validation"],
                        tokenizer=None,
                        compute_metrics=(
                            partial(compute_metrics, std=transformers.y_stds[0])
                            if task_type == "regression"
                            else compute_metrics
                        ),
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
                    )

                    trainer.train()
                    # save final model
                    trainer.save_model(f"{training_args.output_dir}/best_model")

                    # evaluate on test
                    test_score = trainer.evaluate(tokenized["test"])
                    print(f"Test score: {test_score}")
                    with open(f"{training_args.output_dir}/test_score.txt", "w") as f:
                        f.write(str(test_score))
                    wandb.log({"test_score": test_score})
                    wandb.finish()
