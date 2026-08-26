# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from molcrawl.core.paths import CELLXGENE_DATASET_DIR, get_bert_output_path, get_custom_tokenizer_path

# Build the tokenizer using the WordLevel model
from molcrawl.data.rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer

original_tokenizer = TranscriptomeTokenizer()

pre_tokenizer = Tokenizer(WordLevel(vocab=original_tokenizer.gene_token_dict, unk_token="[UNK]"))

# Wrap into Hugging Face tokenizer
tmp_tokenizer = PreTrainedTokenizerFast(tokenizer_object=pre_tokenizer, unk_token="[UNK]", pad_token="[PAD]")
tmp_tokenizer.unk_token = "[UNK]"
tmp_tokenizer.sep_token = "[SEP]"
tmp_tokenizer.pad_token = "<pad>"
tmp_tokenizer.cls_token = "[CLS]"
tmp_tokenizer.mask_token = "<mask>"

_custom_tokenizer_path = get_custom_tokenizer_path("rna", "bert")
tmp_tokenizer.save_pretrained(_custom_tokenizer_path)

tokenizer = AutoTokenizer.from_pretrained(_custom_tokenizer_path)


# 3 epochs of the train split at global batch 2560:
# floor(3 * 34,407,040 train blocks / 2560) = 40,320 steps.
# NOTE: unlike GPT-2/nanoGPT, HF Trainer's effective batch is GPU-count-dependent
# (per_device 8 * grad_accum 80 * world_size), so 2560 only holds on 4 GPUs.
# Running on a different GPU count requires rescaling gradient_accumulation_steps.
max_steps: int = 40320
early_stopping = False  # Pretraining: run the full schedule, no early stopping
# early_stopping_patience: int = 3  # N/A when early_stopping = Falsevement
model_size: str = "small"  # Choose between small, medium or large
model_path: str = get_bert_output_path("rna", model_size)
max_length: int = 1024
dataset_dir: str = CELLXGENE_DATASET_DIR
learning_rate: float  = 0.0001
weight_decay: float  = 0.01
log_interval: int = 100
save_steps: int = 1000  # protein convention; 100 meant ~400 saves over 40,320 steps

# 10 % of max_steps, against the 2 % used elsewhere. The MLM stall investigation
# (tmp/bert-mlm-stall-report-2026-08-06.md) traces the collapse of deep post-LN
# BERT to too short a warmup — the original BERT-large took 10,000 steps, and
# compounds cannot get near that because its whole run is 1,558 steps. RNA's
# 40,320 steps make a real warmup affordable, so it carries the 10 % arm of that
# test (boss decision 2026-08-07, Q3/Q6).
warmup_steps: int = 4032

# Collapse detection is deliberately OFF for RNA. The threshold was 9.3, drawn
# just under the 9.372-nat unigram baseline — but crossing that line is not
# evidence that a run has learned anything. genome's probe made the point: a
# 1e-5 run crossed the degenerate line at step 700 and then learned nothing for
# the next 7,300. The cross-modality rule is to leave detection off until a
# breakout has actually been observed for the modality, and RNA has never had a
# BERT run at all — neither its breakout nor its plateau length is measured.
# Judge these runs by watching the slope of val, not by a level.
#   Aug 2026: the signal that separated stalled from merely slow on genome was
#   the slope, not the level — over the last 2,000 steps a healthy run improved
#   by +0.0330, a slow one by +0.0072, a stalled one by -0.0010. The detector
#   only looks at levels, so using the slope would mean changing it; noted here
#   for whoever next touches CollapseDetectionCallback.

batch_size: int = 8
per_device_eval_batch_size: int = 8
gradient_accumulation_steps: int = 5 * 16

# No preprocess_function here on purpose. training_ready packs cells end-to-end and
# truncates to whole 1024-token blocks, so the data contains no padding at all —
# attention_mask would be all ones and token_type_ids all zeros, exactly what the
# model assumes when they are absent. The old preprocess_function derived the mask
# from `token_id != 0`, which instead zeroed out the EOS separators (id 0, 0.051%
# of positions), hiding them from attention, and it ran a .map() over all
# 34,407,040 train rows to do it.

# Special Tokens
eos_token: int = 0  # eos

# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 106
