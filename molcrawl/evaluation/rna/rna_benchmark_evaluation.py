#!/usr/bin/env python3
"""
RNA Benchmark Evaluation Script

RNAのベンチマークデータ（JSONL）に対して、BERT/GPT-2の評価を行います。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import BertForMaskedLM

from molcrawl.rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer
from molcrawl.utils.evaluation_output import setup_evaluation_logging


@dataclass
class EvaluationConfig:
    """評価設定"""

    model_type: str
    model_path: Path
    data_path: Path
    output_dir: Path
    batch_size: int
    device: str
    max_cells: Optional[int]
    seed: int
    mlm_probability: float


@dataclass
class DatasetSplit:
    """データセットのトークン列"""

    name: str
    tokens: List[List[int]]


def _load_jsonl(data_path: Path, datasets: Optional[List[str]]) -> List[DatasetSplit]:
    """JSONLからデータを読み込み、データセットごとに分ける"""
    selected = {name.strip() for name in datasets} if datasets else None
    grouped: Dict[str, List[List[int]]] = {}

    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            dataset = record["dataset"]
            if selected is not None and dataset not in selected:
                continue
            tokens = record["tokens"]
            grouped.setdefault(dataset, []).append(tokens)

    return [DatasetSplit(name=k, tokens=v) for k, v in grouped.items()]


def _sample_tokens(tokens: List[List[int]], max_cells: Optional[int], seed: int) -> List[List[int]]:
    """細胞数をサンプリングする"""
    if max_cells is None or len(tokens) <= max_cells:
        return tokens
    random.seed(seed)
    return random.sample(tokens, k=max_cells)


def _prepare_bert_inputs(
    tokens_batch: List[List[int]],
    max_length: int,
    pad_token_id: int,
    cls_token_id: int,
    sep_token_id: int,
) -> Tuple[Tensor, Tensor]:
    """BERT用のinput_idsとattention_maskを作成"""
    input_ids_list: List[List[int]] = []
    attention_list: List[List[int]] = []

    for tokens in tokens_batch:
        # CLS/SEPを追加し、最大長に整形
        trimmed = tokens[: max_length - 2]
        seq = [cls_token_id] + trimmed + [sep_token_id]
        attention = [1] * len(seq)
        while len(seq) < max_length:
            seq.append(pad_token_id)
            attention.append(0)
        input_ids_list.append(seq)
        attention_list.append(attention)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_list, dtype=torch.long)
    return input_ids, attention_mask


def _mask_tokens(
    input_ids: Tensor,
    attention_mask: Tensor,
    vocab_size: int,
    pad_token_id: int,
    cls_token_id: int,
    sep_token_id: int,
    mask_token_id: int,
    mlm_probability: float,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    """MLM用にトークンをマスクし、labelsを作成"""
    torch.manual_seed(seed)
    labels = input_ids.clone()

    # マスク対象位置の判定
    special_mask = (input_ids == pad_token_id) | (input_ids == cls_token_id) | (input_ids == sep_token_id)
    probability_matrix = torch.full(labels.shape, mlm_probability, device=labels.device)
    probability_matrix = probability_matrix.masked_fill(special_mask, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels = labels.masked_fill(~masked_indices, -100)

    # 80%: [MASK]に置換
    replace_prob = torch.full(labels.shape, 0.8, device=labels.device)
    replaced = torch.bernoulli(replace_prob).bool() & masked_indices
    input_ids[replaced] = mask_token_id

    # 10%: ランダムトークンに置換
    random_prob = torch.full(labels.shape, 0.5, device=labels.device)
    random_indices = torch.bernoulli(random_prob).bool() & masked_indices & ~replaced
    random_tokens = torch.randint(low=0, high=vocab_size, size=labels.shape, device=labels.device)
    input_ids[random_indices] = random_tokens[random_indices]

    # 10%: そのまま（残り）
    return input_ids, labels


def _evaluate_bert(
    cfg: EvaluationConfig,
    dataset: DatasetSplit,
    logger,
) -> Dict[str, float]:
    """BERT評価（MLM loss）"""
    model = BertForMaskedLM.from_pretrained(cfg.model_path)
    model.to(cfg.device)
    model.eval()

    max_length = model.config.max_position_embeddings

    tokenizer = TranscriptomeTokenizer()
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = tokenizer.gene_token_dict.get(tokenizer.mask_token, 3)
    vocab_size = model.config.vocab_size

    tokens = _sample_tokens(dataset.tokens, cfg.max_cells, cfg.seed)
    total_loss = 0.0
    total_steps = 0

    for i in range(0, len(tokens), cfg.batch_size):
        batch_tokens = tokens[i : i + cfg.batch_size]
        input_ids, attention_mask = _prepare_bert_inputs(
            batch_tokens,
            max_length=max_length,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
        )
        input_ids, labels = _mask_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            mask_token_id=mask_token_id,
            mlm_probability=cfg.mlm_probability,
            seed=cfg.seed + i,
        )
        input_ids = input_ids.to(cfg.device)
        attention_mask = attention_mask.to(cfg.device)
        labels = labels.to(cfg.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        total_loss += loss.item()
        total_steps += 1

        percent = int((i + len(batch_tokens)) / len(tokens) * 100)
        if percent % 1 == 0:
            logger.info(f"Step: evaluation Progress: {percent}% (dataset={dataset.name})")

    avg_loss = total_loss / max(total_steps, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return {"avg_loss": avg_loss, "perplexity": perplexity}


def _load_gpt2_from_checkpoint(model_path: Path):
    """GPT-2（自前）チェックポイントを読み込む"""
    from gpt2.model import GPT, GPTConfig

    training_args_path = model_path / "training_args.json"
    if not training_args_path.exists():
        raise FileNotFoundError(f"training_args.json が見つかりません: {training_args_path}")

    with training_args_path.open("r", encoding="utf-8") as f:
        training_args = json.load(f)

    model_args = training_args.get("model_args", {})
    config = GPTConfig(
        block_size=model_args.get("block_size", 1024),
        vocab_size=model_args.get("vocab_size"),
        n_layer=model_args.get("n_layer", 12),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        dropout=model_args.get("dropout", 0.0),
        bias=model_args.get("bias", True),
    )

    model = GPT(config)
    checkpoint_path = model_path / "pytorch_model.bin"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    return model


def _prepare_gpt2_batch(tokens_batch: List[List[int]], block_size: int, pad_token_id: int) -> Tuple[Tensor, Tensor]:
    """GPT-2用のinput_idsとtargetsを作成"""
    input_ids_list: List[List[int]] = []
    targets_list: List[List[int]] = []

    for tokens in tokens_batch:
        seq = tokens[:block_size]
        if len(seq) < block_size:
            seq = seq + [pad_token_id] * (block_size - len(seq))
        input_seq = seq[:-1]
        target_seq = seq[1:]
        # pad部分をignore_index(-1)にする
        target_seq = [t if t != pad_token_id else -1 for t in target_seq]
        input_ids_list.append(input_seq)
        targets_list.append(target_seq)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    targets = torch.tensor(targets_list, dtype=torch.long)
    return input_ids, targets


def _evaluate_gpt2(
    cfg: EvaluationConfig,
    dataset: DatasetSplit,
    logger,
) -> Dict[str, float]:
    """GPT-2評価（次トークン損失）"""
    model = _load_gpt2_from_checkpoint(cfg.model_path)
    model.to(cfg.device)
    model.eval()

    tokens = _sample_tokens(dataset.tokens, cfg.max_cells, cfg.seed)
    total_loss = 0.0
    total_steps = 0

    block_size = model.config.block_size
    pad_token_id = 0

    for i in range(0, len(tokens), cfg.batch_size):
        batch_tokens = tokens[i : i + cfg.batch_size]
        input_ids, targets = _prepare_gpt2_batch(batch_tokens, block_size=block_size, pad_token_id=pad_token_id)
        input_ids = input_ids.to(cfg.device)
        targets = targets.to(cfg.device)

        with torch.no_grad():
            _, loss = model(input_ids, targets=targets)
        total_loss += loss.item()
        total_steps += 1

        percent = int((i + len(batch_tokens)) / len(tokens) * 100)
        if percent % 1 == 0:
            logger.info(f"Step: evaluation Progress: {percent}% (dataset={dataset.name})")

    avg_loss = total_loss / max(total_steps, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return {"avg_loss": avg_loss, "perplexity": perplexity}


def main() -> None:
    parser = argparse.ArgumentParser(description="RNA Benchmark Evaluation")
    parser.add_argument("--model_type", choices=["bert", "gpt2"], required=True, help="モデル種別")
    parser.add_argument("--model_path", required=True, help="チェックポイントパス")
    parser.add_argument("--data_path", required=True, help="ベンチマークJSONLのパス")
    parser.add_argument("--output_dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--datasets", default="", help="評価対象データセット名（カンマ区切り）")
    parser.add_argument("--batch_size", type=int, default=16, help="評価バッチサイズ")
    parser.add_argument("--device", default="cuda", help="評価デバイス")
    parser.add_argument("--max_cells", type=int, default=None, help="評価対象の最大細胞数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--mlm_probability", type=float, default=0.2, help="BERTのMLM確率")
    args = parser.parse_args()

    cfg = EvaluationConfig(
        model_type=args.model_type,
        model_path=Path(args.model_path),
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        device=args.device,
        max_cells=args.max_cells,
        seed=args.seed,
        mlm_probability=args.mlm_probability,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_evaluation_logging(cfg.output_dir, "rna_benchmark_evaluation")

    datasets = [name for name in args.datasets.split(",") if name.strip()]
    splits = _load_jsonl(cfg.data_path, datasets if datasets else None)
    if not splits:
        raise ValueError("評価対象のデータが見つかりませんでした。")

    results: Dict[str, Dict[str, float]] = {}
    detailed_rows: List[Dict[str, Any]] = []

    for split in splits:
        logger.info(f"Evaluating dataset: {split.name} (cells={len(split.tokens)})")
        if cfg.model_type == "bert":
            metrics = _evaluate_bert(cfg, split, logger)
        else:
            metrics = _evaluate_gpt2(cfg, split, logger)

        results[split.name] = metrics
        detailed_rows.append({"dataset": split.name, "avg_loss": metrics["avg_loss"], "perplexity": metrics["perplexity"]})

    results_file = cfg.output_dir / "rna_benchmark_results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    detailed_file = cfg.output_dir / "rna_benchmark_detailed_results.csv"
    with detailed_file.open("w", encoding="utf-8") as f:
        f.write("dataset,avg_loss,perplexity\n")
        for row in detailed_rows:
            f.write(f"{row['dataset']},{row['avg_loss']:.6f},{row['perplexity']:.6f}\n")

    report_file = cfg.output_dir / "rna_benchmark_evaluation_report.txt"
    with report_file.open("w", encoding="utf-8") as f:
        f.write("RNA Benchmark Evaluation Report\n")
        f.write(f"Model type: {cfg.model_type}\n")
        f.write(f"Model path: {cfg.model_path}\n")
        f.write(f"Data path: {cfg.data_path}\n")
        f.write(f"Batch size: {cfg.batch_size}\n")
        f.write(f"Device: {cfg.device}\n\n")
        for name, metrics in results.items():
            f.write(f"[{name}]\n")
            f.write(f"  avg_loss: {metrics['avg_loss']:.6f}\n")
            f.write(f"  perplexity: {metrics['perplexity']:.6f}\n\n")

    logger.info("=== RNA Benchmark Evaluation Completed ===")
    logger.info(f"Results JSON: {results_file}")
    logger.info(f"Detailed CSV: {detailed_file}")
    logger.info(f"Report TXT: {report_file}")


if __name__ == "__main__":
    main()
