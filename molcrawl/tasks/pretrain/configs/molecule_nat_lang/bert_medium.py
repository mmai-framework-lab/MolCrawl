# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import os

from molcrawl.core.paths import get_bert_output_path
from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer
from molcrawl.data.molecule_nat_lang.utils.vocab_guard import check_vocab_size

# Get LEARNING_SOURCE_DIR from environment variable directly
LEARNING_SOURCE_DIR = os.environ.get("LEARNING_SOURCE_DIR", "./learning_source_20260105-molecule-nl")
MOLECULE_NAT_LANG_DIR = LEARNING_SOURCE_DIR + "/molecule_nat_lang"
MOLECULE_NAT_LANG_DATASET_DIR = MOLECULE_NAT_LANG_DIR + "/training_ready_hf_dataset"

tokenizer = Tokenizer()

# molecule_nat_lang uses the GPT-2 tokenizer (vocab_size=50257). Pad up to
# the next multiple of 8 for efficient embedding lookups. check_vocab_size()
# verifies the result matches the value baked into existing checkpoints so
# a tokenizer swap is caught at startup rather than silently trashing weights.
meta_vocab_size = (tokenizer.vocab_size // 8 + 1) * 8
check_vocab_size(meta_vocab_size)

# 12,000 steps at effective global batch 2,560 -- the value the learning-rate grid of
# 2026-09-15 is run at, and the length the small, medium and large arms are compared at.
# The 373 this replaces was 3 epochs of the train split (3 * 318,118 / 2,560), which the
# runs up to 2026-09-13 overrode at launch to 12,000 anyway; written here so the grid
# arms carry it in the config rather than in the submit command.
max_steps = 12000
# 10% of max_steps. models/bert/main.py sets 200 when a config says nothing, which is the
# value shared by five modalities; this config states its own rather than inheriting it.
warmup_steps = 1200
# MLM collapse fix: packing concatenates ~10 documents per 1024 block (measured
# 10.26 EOS per block on the train split); without masking, attention leaks across
# those documents and the run stalls at the unigram level. Confine attention per
# document. Requires the tokenizer to expose EOS as sep_token (see
# data/molecule_nat_lang/utils/tokenizer.py).
document_masking = True

early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "medium"  # Choose between small, medium or large
model_path = get_bert_output_path("molecule_nat_lang", model_size)
max_length = 1024
# Shuffled rebuild, the same corpus the GPT-2 ladder switched to in #143. The
# original was written in source order: JS divergence between the head and the
# middle of train measured 0.16449 against a 0.00101 sampling floor, and the
# rebuild brings it to 0.00096. Content is identical (325,752,832 tokens,
# 3,267,172 documents); only the grouping into 1024-token blocks differs.
dataset_dir = MOLECULE_NAT_LANG_DATASET_DIR + "_shuffled"
# Provisional: the 2026-09-15 grid runs 1e-4 / 3e-4 / 1e-3 at each size in the
# bert_*_lr*.py configs, and the winner gets written back here.
learning_rate = 1e-4
weight_decay = 0.01
log_interval = 100
save_steps = 1000  # Save checkpoint every 1000 steps instead of 100

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16
# Where the input is fetched, not what is computed: the same rows in the same
# order, pulled by four worker processes into pinned buffers instead of by the
# training process itself. main.py defaults both off (main.py:655-656), and with
# them off the Arrow read, the MLM draw and the document masking all sit on the
# critical path of every step. Measured on rna small at 8 x 80, 4 GPUs: in fp32
# these two settings alone take a step from 12.476 to 11.044 s (1.13x); with bf16
# as well it falls to 3.812 s (3.27x from where it started). bf16 without them is
# 0.94x. Neither half does much on its own.
#
# The masked positions are not the same as a 0-worker run: the collate runs in
# the worker, whose RNG PyTorch seeds per worker. The rate and the objective are
# unchanged and each setting reproduces itself, but the draw differs, so a run
# started with workers is not a continuation of one started without.
#
# Approved 2026-09-16 (all-bert-throughput-verdict-2026-09-16b). bf16 was not:
# on its own it measured 0.94x, and it changes numerics rather than placement.
dataloader_num_workers = 4
dataloader_pin_memory = True

# The number max_steps was derived from, stated so the run can check it rather
# than assume it. main.py multiplies per_device x grad_accum x world_size at
# startup and refuses to train if the product differs: under HF the effective
# batch moves with the GPU count, and a 4-GPU request that the scheduler splits
# across 2 nodes would silently train at 1,280.
expected_global_batch = 2560


# Add preprocessing function to create attention_mask
def preprocess_function(examples):
    """Add attention_mask to the dataset"""
    if "input_ids" in examples:
        # Create attention_mask: 1 for real tokens, 0 for padding
        attention_masks = []
        for input_ids in examples["input_ids"]:
            # Assuming pad_token_id is 0
            attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
            attention_masks.append(attention_mask)

        examples["attention_mask"] = attention_masks

    return examples



# 42 across all three sizes, so the learning-rate grid differs in the learning rate alone.
# This leaves the per-config sequential seeds assigned on 2026-08-03 (small 59, medium 55,
# large 54), which the runs up to 2026-09-13 used; those runs are kept as reference values
# and are not part of the grid. Seed variance is therefore not measured here.
seed = 42

# Evaluation runs every 100 steps and checkpoints are written every 1,000, so of the 120
# eval points only 12 leave weights behind: the best number reported and the weights that
# can be adopted would come from different steps. Ask for a checkpoint at each new best as
# well (models/bert/_save_on_improve).
save_on_improve = True
