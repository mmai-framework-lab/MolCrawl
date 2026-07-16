# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

max_steps = 60000
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "xl"  # Choose between small, medium or large
model_path = get_bert_output_path("compounds", model_size)
max_length = 128
dataset_dir = COMPOUNDS_DATASET_DIR
learning_rate = 0.0001
weight_decay = 0.01
log_interval = 100

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 10  # 80/8GPU = 10 (effective batch 640)


# === XL-scale speedups (added 2026-06-25 for the 5-modality XL campaign) ===
# These are read by molcrawl/models/bert/main.py via globals().get(...),
# so existing configs that don't set them stay on the legacy defaults.
bf16 = True                            # mixed precision (Hopper / Blackwell / MI300X)
tf32 = True                            # TF32 matmuls on Ampere+
dataloader_num_workers = 4
dataloader_pin_memory = True
dataloader_persistent_workers = True   # skip per-epoch worker respawn
flash_attention = True                 # _attn_implementation = "flash_attention_2"
torch_compile = True                   # inductor kernel fusion
optim = "adamw_torch_fused"            # fused CUDA AdamW
ddp_bucket_cap_mb = 512                # larger AllReduce buckets (HF default 25)
