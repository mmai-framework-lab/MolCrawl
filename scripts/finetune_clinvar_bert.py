"""Fine-tune a subset BERT pretrain checkpoint on ClinVar variant
pathogenicity classification, then report AUROC on a held-out
chromosome split.

Rationale
---------
The zero-shot PLL ClinVar AUROC (see scripts/run_clinvar_eval_subsets.py)
plateaued around chance level on high-quality (≥2★) variants on our
small models with single-nucleotide tokenizer. This script tests the
hypothesis that *fine-tuning* — adding a small classification head
and updating the trunk on a labelled ClinVar train split — recovers
useful signal.

The script is BERT-only by design (HuggingFace
``BertForSequenceClassification`` lets us re-use the trunk and add a
fresh 2-class head with one line). A separate GPT-2 implementation
will follow once nanoGPT-format trunks have been wrapped for
classification.

Data split policy
-----------------
We use a *chromosome-based* train / test split rather than a random
row-level shuffle. This avoids the easy leakage where two ClinVar
variants from the same gene end up on opposite sides of the split.
Default: test = ``("chr8", "chrX", "chrY")``, train = the rest.

If the input CSV carries ``vcv_id`` (the
``feat/clinvar-csv-vcv-metadata`` schema), the script logs every
train / test vcv_id to ``splits_manifest.csv`` so the resulting
metrics can be reproduced exactly.

Usage
-----
    python scripts/finetune_clinvar_bert.py \
        --model-path     $LEARNING_SOURCE_DIR/genome_sequence/bert-output/.../checkpoint-60000 \
        --tokenizer-path $LEARNING_SOURCE_DIR/genome_sequence/custom_tokenizer_bert_single_nuc \
        --clinvar-data   $LEARNING_SOURCE_DIR/genome_sequence/clinvar/clinvar_sequences.csv \
        --output-dir     $LEARNING_SOURCE_DIR/genome_sequence/analysis/clinvar_finetune/bert__<subset>__seed1 \
        --num-train-steps 500 \
        --learning-rate 1e-5 \
        --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("finetune_clinvar_bert")


DEFAULT_TEST_CHROMS = ("8", "X", "Y")
PATHOGENIC_TERMS = ["pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"]
BENIGN_TERMS = ["benign", "likely benign", "benign/likely benign"]


def label_of(value) -> int | None:
    if pd.isna(value):
        return None
    s = str(value).lower()
    if any(t in s for t in PATHOGENIC_TERMS):
        return 1
    if any(t in s for t in BENIGN_TERMS):
        return 0
    return None


def load_clinvar_for_finetune(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["chrom"] = df["chrom"].astype(str)
    df["pathogenic"] = df["ClinicalSignificance"].apply(label_of)
    df = df.dropna(subset=["pathogenic", "variant_sequence"]).reset_index(drop=True)
    df["pathogenic"] = df["pathogenic"].astype(int)
    return df


def chromosome_split(
    df: pd.DataFrame, test_chroms: Tuple[str, ...]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_set = {c.lower().removeprefix("chr") for c in test_chroms}
    chrom = df["chrom"].str.lower().str.removeprefix("chr")
    test_mask = chrom.isin(test_set)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def write_splits_manifest(
    out_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    rows: List[dict] = []
    for split_name, df in (("train", train_df), ("test", test_df)):
        for _, row in df.iterrows():
            rows.append({
                "split": split_name,
                "vcv_id": row.get("vcv_id"),
                "chrom": row.get("chrom"),
                "pos": row.get("pos"),
                "ref": row.get("ref"),
                "alt": row.get("alt"),
                "pathogenic": int(row["pathogenic"]),
                "review_status": row.get("review_status"),
                "consequence": row.get("consequence"),
            })
    fields = ["split", "vcv_id", "chrom", "pos", "ref", "alt",
              "pathogenic", "review_status", "consequence"]
    out_path = out_dir / "splits_manifest.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info(
        "wrote splits manifest: %s (train=%d, test=%d)",
        out_path, len(train_df), len(test_df),
    )


def balance_classes(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Down-sample the majority class to match the minority count."""
    pos = df[df["pathogenic"] == 1]
    neg = df[df["pathogenic"] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        return df.iloc[0:0]
    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(pos.index, size=n, replace=False)
    neg_idx = rng.choice(neg.index, size=n, replace=False)
    return pd.concat([pos.loc[pos_idx], neg.loc[neg_idx]]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, help="Pretrained ckpt dir")
    ap.add_argument("--tokenizer-path", required=True,
                    help="HF tokenizer dir (custom_tokenizer_bert_single_nuc)")
    ap.add_argument("--clinvar-data", required=True, help="ClinVar CSV (sequences + label)")
    ap.add_argument("--output-dir", required=True, help="Where metrics.json / predictions.jsonl land")
    ap.add_argument("--test-chroms", default=",".join(DEFAULT_TEST_CHROMS),
                    help='Comma-separated chromosome list for the held-out test split (e.g. "8,X,Y")')
    ap.add_argument("--num-train-steps", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--per-device-batch-size", type=int, default=16)
    ap.add_argument("--max-train-rows", type=int, default=20_000,
                    help="Cap on train rows after class balancing (per side: max/2)")
    ap.add_argument("--max-test-rows", type=int, default=4_000,
                    help="Cap on test rows after class balancing (per side: max/2)")
    ap.add_argument("--max-length", type=int, default=192,
                    help="Tokenizer max_length for the variant_sequence (~128 nt + special)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments,
        DataCollatorWithPadding,
    )
    from datasets import Dataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("loading ClinVar from %s", args.clinvar_data)
    df = load_clinvar_for_finetune(args.clinvar_data)
    logger.info(
        "loaded %d rows (pathogenic=%d, benign=%d)",
        len(df), int((df["pathogenic"] == 1).sum()), int((df["pathogenic"] == 0).sum()),
    )

    test_chroms = tuple(c.strip() for c in args.test_chroms.split(",") if c.strip())
    train_df, test_df = chromosome_split(df, test_chroms)
    logger.info(
        "chromosome split (test=%s) → train=%d / test=%d",
        test_chroms, len(train_df), len(test_df),
    )

    train_df = balance_classes(train_df, seed=args.seed)
    test_df  = balance_classes(test_df, seed=args.seed)
    if args.max_train_rows and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(n=args.max_train_rows, random_state=args.seed).reset_index(drop=True)
    if args.max_test_rows and len(test_df) > args.max_test_rows:
        test_df = test_df.sample(n=args.max_test_rows, random_state=args.seed).reset_index(drop=True)
    logger.info("after balance + cap: train=%d / test=%d", len(train_df), len(test_df))

    write_splits_manifest(out_dir, train_df, test_df)

    logger.info("loading tokenizer from %s", args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logger.info("vocab_size=%d", tokenizer.vocab_size)

    logger.info("loading BERT from %s", args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=2,
    )

    def tokenize(batch):
        return tokenizer(
            batch["variant_sequence"], truncation=True, max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["variant_sequence", "pathogenic"]].rename(columns={"pathogenic": "labels"}))
    test_ds  = Dataset.from_pandas(test_df[["variant_sequence", "pathogenic"]].rename(columns={"pathogenic": "labels"}))
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["variant_sequence"])
    test_ds  = test_ds.map(tokenize, batched=True, remove_columns=["variant_sequence"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=10,  # capped via max_steps below
        max_steps=args.num_train_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=max(1, args.num_train_steps // 20),
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=args.seed,
        fp16=(args.device == "cuda"),
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=test_ds,
        tokenizer=tokenizer, data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60.0
    logger.info("fine-tune done in %.1f min", train_min)

    logger.info("predicting on test split (n=%d)", len(test_ds))
    pred_out = trainer.predict(test_ds)
    logits = pred_out.predictions
    labels = pred_out.label_ids
    probs_pos = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs_pos > 0.5).astype(int)

    auroc = float(roc_auc_score(labels, probs_pos))
    auprc = float(average_precision_score(labels, probs_pos))
    acc   = float((preds == labels).mean())
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    metrics = {
        "auroc": auroc, "auprc": auprc, "accuracy": acc,
        "n_test": int(len(labels)), "n_pos": n_pos, "n_neg": n_neg,
        "train_minutes": train_min,
    }
    details = {
        "task": "clinvar_finetune",
        "modality": "genome_sequence",
        "arch": "bert",
        "model_path": str(args.model_path),
        "tokenizer_path": str(args.tokenizer_path),
        "config": {
            "num_train_steps": args.num_train_steps,
            "learning_rate": args.learning_rate,
            "per_device_batch_size": args.per_device_batch_size,
            "max_length": args.max_length,
            "test_chroms": list(test_chroms),
            "seed": args.seed,
        },
        "n_train": int(len(train_ds)),
        "n_test": int(len(test_ds)),
        "metrics": metrics,
    }
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump({"metrics": metrics, "details": details}, fh, indent=2)

    # Per-variant predictions (lean JSONL)
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w") as fh:
        for i, row in test_df.iterrows():
            rec = {
                "vcv_id": row.get("vcv_id"),
                "chrom": row.get("chrom"),
                "pos": int(row.get("pos")) if not pd.isna(row.get("pos")) else None,
                "ref_allele": row.get("ref"), "alt_allele": row.get("alt"),
                "review_status": row.get("review_status"),
                "consequence": row.get("consequence"),
                "label_pathogenic": int(labels[i]),
                "pred_prob": float(probs_pos[i]),
                "pred_pathogenic": int(preds[i]),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n=== {args.output_dir} ===")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
