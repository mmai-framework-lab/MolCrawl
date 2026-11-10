# DIAGNOSTIC config (untracked): protein bert-small with the ESM-standard MLM optimizer.
#
# Root-cause test (2026-08-25). Production protein bert-small (run 24870) collapses to the
# unigram marginal. Work-1 showed the SAME BertForMaskedLM learns protein MLM when run under
# the esm2 pipeline (eval 2.67->2.56). The only optimizer differences between the collapsing
# bert/main.py recipe and the learning esm2 recipe are:
#     adam_beta2 : 0.95 (GPT-2 production spec) -> 0.999 (HF/BERT/ESM standard)
#     warmup_steps: 200 -> 2000
# This config is bert_small.py with ONLY those two changed, so a run here vs 24870 isolates the
# optimizer bundle. Learns -> optimizer was the culprit AND this IS the shippable small recipe.
#
# Everything else (LR 3e-4, eff-batch 2560, max_steps, document_masking, threshold, seed) is
# identical to production bert_small.py.

from typing import Any, Dict, List

import torch
from transformers import DataCollatorForLanguageModeling

from molcrawl.core.paths import UNIPROT_DATASET_DIR, get_bert_output_path
from molcrawl.data.protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer

tokenizer = create_bert_protein_tokenizer()


def preprocess_function(examples):
    if "input_ids" in examples:
        input_ids = examples["input_ids"]
        if isinstance(input_ids[0], list):
            attention_masks = []
            for seq in input_ids:
                pad_token_id = (
                    tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
                )
                attention_masks.append([1 if token != pad_token_id else 0 for token in seq])
            examples["attention_mask"] = attention_masks
        else:
            pad_token_id = (
                tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
            )
            examples["attention_mask"] = [1 if token != pad_token_id else 0 for token in input_ids]
    return examples


class ProteinSequenceDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        for example in examples:
            if "sequence_tokens" in example and "input_ids" not in example:
                example["input_ids"] = example.pop("sequence_tokens")
        return super().torch_call(examples)


data_collator = ProteinSequenceDataCollator(tokenizer=tokenizer, mlm=True, mlm_probability=0.40)  # CASE C (boss 2026-08-31): train mask 0.20->0.40 (protein has the smallest gap to the context-free solution; raise teacher signal per step). eval compares at 0.20 via post-hoc re-score.

max_steps = 11177  # 3 epochs of train at global batch 2560 = floor(3*9,538,464/2560)
early_stopping = False
document_masking = True
degenerate_loss_threshold = None  # detector OFF for protein (boss 2026-08-26)
model_size = "small"
model_path = get_bert_output_path("protein_sequence", model_size)  # overridden via --model_path on launch
max_length = 1024
dataset_dir = UNIPROT_DATASET_DIR
learning_rate = 0.0003  # unchanged from production
weight_decay = 0.01
log_interval = 100
save_steps = 1000

# Keep the checkpoint the reported number came from. Evaluation is 10x finer than
# saving here (100 against 1,000), so the minimum lands off the save grid nine
# times in ten and best_model_checkpoint points at a neighbour instead.
save_on_improve = True

# --- THE ONLY CHANGES vs bert_small.py: ESM-standard MLM optimizer ---
adam_beta2 = 0.999   # was 0.95 (GPT-2 spec); ESM/BERT MLM standard
warmup_steps = 2000  # was 200; matches esm2/main.py
# --------------------------------------------------------------------

batch_size = 160  # throughput 160x4 (genome finding); eff-batch 2560 unchanged
# Match the training micro-batch. Evaluation reads a fixed 10,000 rows
# (models/bert/main.py EVAL_SUBSET_ROWS) however it is batched, so 8 against a
# training 160 only makes each eval point ~20x slower in wall time -- the shape
# genome measured at 8.08 s per eval, 16.3 % of the run at eval_interval=100, and
# widened its interval to 1,000 to escape (4972a37). This config runs
# eval_interval=100, so it would pay that in full. Safe: evaluation runs two
# no_grad forwards per batch (HF's own and the breakdown's in _mlm_diagnostics)
# and two of those peak below the one forward+backward training already does at
# this width; logits are 160 x 1,024 x 33 vocab = 22 MB.
per_device_eval_batch_size = 160
gradient_accumulation_steps = 4  # eff-batch 160*4GPU*4 = 2560

meta_vocab_size = len(tokenizer.get_vocab())
# Training seed 42, as every config in this tree declares (igarashi's order of
# 2026-09-30, applied across the tree on 2026-11-10, and to this file on 2026-11-11).
# The diagnostic runs launched from it went at seed 82; that value is in their
# training_args.bin and in docs/07-reports/protein_bert_mlm_diagnostic_configs.md.
seed = 42
