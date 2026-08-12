"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import glob
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from molcrawl.core.dataset import PreparedDataset

# Checkpoints carry RNG state, and numpy's includes arrays that torch >= 2.6
# refuses to unpickle under its new weights_only=True default. Restore the old
# default before any resume reads one. See molcrawl/core/torch_compat; the BERT
# and RoBERTa trainers already do the same.
from molcrawl.core.torch_compat import enable_full_torch_load
from molcrawl.data.rna.dataset.rna_dataset import RNADataset
from molcrawl.models.gpt2.model import GPT, GPTConfig

# Call after the imports, not between them, so nothing here trips E402. Order
# does not matter beyond running before the first torch.load, which happens
# inside the training entry point far below.
enable_full_torch_load()

dataset_params: dict[str, object] = {}
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

tensorboard = False  # log training metrics to tensorboard
tensorboard_dir = "runs"
# wandb settings (can be overridden by environment variables, config file, or command line args)
use_wandb = os.environ.get("USE_WANDB", "False").lower() in ("true", "1", "yes")  # log training metrics to wandb
wandb_project = os.environ.get("WANDB_PROJECT", "gpt2-training")  # wandb project name
wandb_run_name = os.environ.get("WANDB_RUN_NAME", None)  # wandb run name (None = auto-generate)
wandb_entity = os.environ.get("WANDB_ENTITY", None)  # wandb entity/team name (None = default)
wandb_log_model = os.environ.get("WANDB_LOG_MODEL", "False").lower() in (
    "true",
    "1",
    "yes",
)  # log model checkpoints as wandb artifacts

out_dir = "out-gpt2"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
pretrain_dir = ""  # Path to pretraining out_dir; used when init_from='resume' and out_dir has no checkpoint
# checkpoint management (Hugging Face style)
save_hf_checkpoints = True  # if True, save checkpoints in HF format (checkpoint-{step}/)
save_checkpoint_steps = None  # save checkpoint every N steps (None = use eval_interval)
max_checkpoints = 3  # maximum number of checkpoints to keep (None = keep all)
keep_legacy_ckpt = False  # if True, also save ckpt.pt for backward compatibility
# early stopping
early_stopping = False  # if True, stop training when validation loss stops improving
early_stopping_patience = 10  # number of eval_intervals to wait before stopping
# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 60000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 60000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
# MFU reference: accelerator bf16 dense peak in TFLOPS, used only to report MFU
# as a fraction of hardware peak. Default ~ NVIDIA GB200/B200 bf16 dense; set
# --mfu_peak_tflops=312 for A100, or your measured value for an exact ratio.
mfu_peak_tflops = 2250
# Training seed (base). Overridden by the configurator when a config file sets
# `seed = N`. Consumed below to seed torch / torch.cuda / numpy / Python
# random; per-rank offset (seed + ddp_rank) is applied so DDP ranks still draw
# distinct mini-batches. 1337 kept as the base to match the historical value
# used before the config-level seed was introduced.
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
# RNG helpers. Defined here, ahead of the first `if __name__` block, because
# the resume path in the second one calls them: this module runs top to
# bottom, so a definition placed after that block is not bound yet when it
# executes (NameError on resume, which CI caught as F821/used-before-def).
def capture_rng_state():
    """Snapshot every RNG that affects training, for exact resume.

    Without this a resumed run diverges from an uninterrupted one even with a
    fixed seed, because the streams restart at ``seed + rank`` instead of
    continuing. ``get_batch`` draws its indices from ``np.random``, so numpy is
    what governs data order; torch/cuda cover dropout and any other sampling.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    """Restore a snapshot from :func:`capture_rng_state`; tolerate old checkpoints.

    Returns True if anything was restored. Checkpoints written before this was
    added simply carry no state, in which case the run continues from the
    freshly seeded streams (the previous behaviour) rather than failing.
    """
    if not state:
        return False
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        # Only restore when the device count matches; a different topology
        # would otherwise raise deep inside torch.
        if len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_state])
    return True


def rank_rng_path(checkpoint_dir, rank):
    return os.path.join(checkpoint_dir, f"rng_state_{rank}.pth")

if __name__ == "__main__":
    # Handle configurator path (support repo-root invocation and direct invocation)
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(_this_dir, "configurator.py")):
        configurator_path = os.path.join(_this_dir, "configurator.py")
    elif os.path.exists("src/gpt2/configurator.py"):
        configurator_path = "src/gpt2/configurator.py"
    elif os.path.exists("gpt2/configurator.py"):
        configurator_path = "gpt2/configurator.py"
    else:
        configurator_path = "configurator.py"
    exec(open(configurator_path).read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    # create folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # create empty csv file for logging with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_file = os.path.join(out_dir, f"logging_{timestamp}.csv")

    with open(logging_file, "w") as f:
        f.write("iter, train_loss, val_loss\n")

    writer = None
    if tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(tensorboard_dir)

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # Extend the NCCL collective timeout well past the default 10 min: a
        # rank-0 checkpoint save to Lustre can occasionally stall other ranks
        # at the next collective long enough to trip the watchdog (observed
        # killing a 4-GPU run mid-training at a checkpoint boundary).
        init_process_group(backend=backend, timeout=timedelta(minutes=30))
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    # Initialize wandb if enabled
    wandb_run = None
    if use_wandb and master_process:
        import wandb

        # Generate run name if not provided
        if wandb_run_name is None:
            wandb_run_name = f"{dataset}-{timestamp}"

        # Add metadata tags for experiment management
        tags = ["gpt2", "training", dataset]

        # Add experiment metadata to config
        experiment_config = {
            **config,
            "experiment_type": "training",
            "model_type": "gpt2",
            "dataset_type": dataset,
        }

        # Initialize wandb
        wandb_run = wandb.init(  # type: ignore[attr-defined]
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=experiment_config,
            tags=tags,
            resume="allow",  # Allow resuming if run exists
        )
        print(f"Wandb initialized: {wandb_run.url}")

    # Full seed control (boss directive 2026-08-03). Config `seed` overrides
    # the 1337 default via the configurator; per-rank offset preserves the
    # DDP behaviour of distinct mini-batches per rank.
    _rank_seed = seed + seed_offset
    torch.manual_seed(_rank_seed)
    torch.cuda.manual_seed_all(_rank_seed)
    np.random.seed(_rank_seed)
    random.seed(_rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Resolve ambiguous token ids once for CLM-side loss masking (legacy
    # ablation path; same policy as the Llama-style train loop).
    from molcrawl.models._collators import (
        ambiguous_tokens_for_modality,
        resolve_ambiguous_token_ids,
    )
    _ambig_tokens = ambiguous_tokens_for_modality(dataset)
    _tokenizer = globals().get("tokenizer")
    if _ambig_tokens and _tokenizer is not None:
        ambiguous_token_ids = resolve_ambiguous_token_ids(_tokenizer, _ambig_tokens)
    else:
        ambiguous_token_ids = []

    # Resolve pad-token id to be excluded from CLM loss. Config may set
    # ``pad_token_id_for_loss = 0`` (compounds convention) so that pad
    # positions produced by the batch padder or already baked into the
    # dataset are ignored during cross_entropy. Default None → no mask
    # (safe for modalities where the pad id value collides with a real
    # token, e.g. genome single-nucleotide vocab where id=0 is 'A').
    pad_token_id_for_loss = globals().get("pad_token_id_for_loss", None)

    # RNA data loader - get paths from config if available
    rna_data_dir = globals().get("rna_data_dir", "path-to-rna-parquet")
    rna_vocab_file = globals().get("rna_vocab_file", "path-to-rna-vocab")

    # Use RNADataset if dataset is "rna", otherwise use PreparedDataset
    if dataset == "rna":
        training_data = RNADataset(rna_data_dir, split="train", vocab_file=rna_vocab_file, test_size=0.1)
        test_data = RNADataset(rna_data_dir, split="valid", vocab_file=rna_vocab_file, test_size=0.1)
        # Set vocab size from the RNA dataset
        meta_vocab_size = training_data.vocab_size
    else:
        print(f"Loading dataset: {dataset_params}")
        training_data = PreparedDataset(**dataset_params, split="train")  # type: ignore[assignment]
        test_data = PreparedDataset(**dataset_params, split="valid")  # type: ignore[assignment]

# training_data = torch.load(os.path.join(data_dir, "train.pt"))
# test_data = torch.load(
# os.path.join(data_dir, "train.pt")
# )  # For now, we use the training data for validation as we want to overfit to confirm the model is working with the data


def get_batch(split):
    if split == "train":
        data = training_data
    elif split == "val":
        data = test_data

    ix = np.random.randint(0, len(data), batch_size).tolist()

    # Handle variable length sequences for RNA data
    sequences = [data[i] for i in ix]

    # Pad or truncate sequences to block_size
    padded_sequences = []
    for seq in sequences:
        # Ensure seq is long type for embedding layer compatibility
        seq = seq.long() if seq.dtype != torch.long else seq

        if len(seq) > block_size:
            # Truncate to block_size
            padded_sequences.append(seq[:block_size])
        elif len(seq) < block_size:
            # Pad with zeros (assuming 0 is padding token)
            padding = torch.zeros(block_size - len(seq), dtype=torch.long)
            padded_sequences.append(torch.cat([seq, padding]))
        else:
            padded_sequences.append(seq)

    batch = torch.stack(padded_sequences)
    x = batch[:, :-1].long()  # Ensure long type for embedding
    y = batch[:, 1:].long()  # Ensure long type for embedding

    # Zero out CLM loss at positions whose target is an ambiguous token
    # (modality-specific; e.g. genome N + IUPAC codes). No-op when the list
    # is empty. Applied to the shifted target y, not the input x.
    if ambiguous_token_ids:
        from molcrawl.models._collators import mask_ambiguous_targets_for_clm
        y = mask_ambiguous_targets_for_clm(y, ambiguous_token_ids)

    # Zero out CLM loss at pad-token positions (target == pad_token_id_for_loss).
    # Applied only when config explicitly sets ``pad_token_id_for_loss`` — for
    # compounds this is 0. Reuses the same helper so ambiguous + pad masks stack.
    if pad_token_id_for_loss is not None:
        from molcrawl.models._collators import mask_ambiguous_targets_for_clm
        y = mask_ambiguous_targets_for_clm(y, [pad_token_id_for_loss])

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
if __name__ == "__main__":
    iter_num = 0
    best_val_loss = 1e9
    best_val_step = None  # step at which best_val_loss was recorded (charter § 1.1 protect)
    early_stopping_counter = 0  # count eval intervals without improvement

    if not ("meta_vocab_size" in vars() and "meta_vocab_size" in globals()):
        if dataset == "rna":
            # meta_vocab_size already set from RNADataset
            pass
        else:
            # For non-RNA datasets, meta_vocab_size should be set in config
            if "meta_vocab_size" not in globals():
                raise ImportError(
                    "Please initialize the variable meta_vocab_size in the *_config.py file with the size of your vocabulary."
                )

    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
    )  # start with model_args from command line

    checkpoint = None  # Initialize checkpoint variable
    _resume_ckpt_dir = None  # dir the resume checkpoint came from (for per-rank RNG)

    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)  # type: ignore[arg-type]
        model = GPT(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        # Try new HuggingFace format first (checkpoint-{step}/training_state.bin)
        # Fall back to legacy format (ckpt.pt)
        checkpoint_loaded = False

        # Find the latest checkpoint directory
        checkpoint_dirs = glob.glob(os.path.join(out_dir, "checkpoint-*"))
        if checkpoint_dirs:
            # Sort by step number to find the latest
            checkpoint_steps = []
            for ckpt_dir in checkpoint_dirs:
                try:
                    step = int(os.path.basename(ckpt_dir).split("-")[1])
                    training_state_path = os.path.join(ckpt_dir, "training_state.bin")
                    if os.path.exists(training_state_path):
                        checkpoint_steps.append((step, training_state_path))
                except (ValueError, IndexError):
                    continue

            if checkpoint_steps:
                checkpoint_steps.sort(reverse=True)
                latest_step, latest_ckpt_path = checkpoint_steps[0]
                print(f"Found HuggingFace format checkpoint at step {latest_step}")
                checkpoint = torch.load(latest_ckpt_path, map_location=device)
                checkpoint_loaded = True
                _resume_ckpt_dir = os.path.dirname(latest_ckpt_path)

        # Fall back to legacy ckpt.pt
        if not checkpoint_loaded:
            ckpt_path = os.path.join(out_dir, "ckpt.pt")
            if os.path.exists(ckpt_path):
                print(f"Loading legacy checkpoint from {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=device)
                checkpoint_loaded = True

        if not checkpoint_loaded:
            print(f"No checkpoint found in {out_dir}")
            # Try to load from pretrain_dir if specified
            if pretrain_dir and os.path.exists(pretrain_dir):
                print(f"Attempting to load pretraining checkpoint from {pretrain_dir}")
                _pretrain_checkpoint_loaded = False
                # Try HuggingFace format first
                _pretrain_ckpt_dirs = glob.glob(os.path.join(pretrain_dir, "checkpoint-*"))
                if _pretrain_ckpt_dirs:
                    _pretrain_steps = []
                    for _d in _pretrain_ckpt_dirs:
                        try:
                            _s = int(os.path.basename(_d).split("-")[1])
                            _ts = os.path.join(_d, "training_state.bin")
                            if os.path.exists(_ts):
                                _pretrain_steps.append((_s, _ts))
                        except (ValueError, IndexError):
                            continue
                    if _pretrain_steps:
                        _pretrain_steps.sort(reverse=True)
                        _, _pretrain_ckpt_path = _pretrain_steps[0]
                        print(f"Loading pretraining checkpoint from {_pretrain_ckpt_path}")
                        checkpoint = torch.load(_pretrain_ckpt_path, map_location=device)
                        _pretrain_checkpoint_loaded = True
                # Fall back to legacy ckpt.pt
                if not _pretrain_checkpoint_loaded:
                    _legacy = os.path.join(pretrain_dir, "ckpt.pt")
                    if os.path.exists(_legacy):
                        print(f"Loading pretraining legacy checkpoint from {_legacy}")
                        checkpoint = torch.load(_legacy, map_location=device)
                        _pretrain_checkpoint_loaded = True
                if not _pretrain_checkpoint_loaded:
                    raise ValueError(
                        f"pretrain_dir={pretrain_dir!r} is set but no checkpoint was found inside it.\n"
                        "Run pretraining first, or unset pretrain_dir to train from scratch."
                    )
            else:
                if pretrain_dir:
                    raise ValueError(
                        f"pretrain_dir={pretrain_dir!r} is set but does not exist.\n"
                        "Run pretraining first, or unset pretrain_dir to train from scratch."
                    )
            if checkpoint is None:
                print("No checkpoint found and pretrain_dir not set — starting training from scratch.")
                # Initialize from scratch if checkpoint doesn't exist
                if meta_vocab_size is None:
                    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
                model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
                gptconf = GPTConfig(**model_args)  # type: ignore[arg-type]
                model = GPT(gptconf)
            else:
                # Pretrain checkpoint loaded via pretrain_dir — use same loading path
                # as the out_dir resume branch below.
                checkpoint_model_args = checkpoint["model_args"]
                for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                    model_args[k] = checkpoint_model_args[k]
                gptconf = GPTConfig(**model_args)  # type: ignore[arg-type]
                model = GPT(gptconf)
                state_dict = checkpoint["model"]
                unwanted_prefix = "_orig_mod."
                for k, _v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
                model.load_state_dict(state_dict)
                # Reset training counters — we are fine-tuning, not resuming
                iter_num = 0
                best_val_loss = 1e9
                best_val_step = None
                early_stopping_counter = 0
                print("Pretraining weights loaded. Starting fine-tuning from iter 0.")
        else:
            checkpoint_model_args = checkpoint["model_args"]
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**model_args)  # type: ignore[arg-type]
            model = GPT(gptconf)
            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k, _v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint["iter_num"]
            best_val_loss = checkpoint["best_val_loss"]
            best_val_step = checkpoint.get("best_val_step")  # may be None for older ckpts
            early_stopping_counter = checkpoint.get("early_stopping_counter", 0)
    elif init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = block_size  # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == "resume" and checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Restore RNG so a chained run continues the same streams instead of
        # replaying from `seed + rank`. Prefer this rank's own file; fall back
        # to the state embedded in the checkpoint, which is the saving rank's
        # (correct for single-process runs). Checkpoints predating this carry
        # neither, and then we keep the previous behaviour rather than fail.
        _rank = ddp_rank if ddp else 0
        _rng_restored = False
        if _resume_ckpt_dir:
            _rng_file = rank_rng_path(_resume_ckpt_dir, _rank)
            if os.path.exists(_rng_file):
                _rng_restored = restore_rng_state(torch.load(_rng_file, map_location="cpu"))
        if not _rng_restored:
            _rng_restored = restore_rng_state(checkpoint.get("rng_state"))
            if _rng_restored and ddp and ddp_world_size > 1:
                print(
                    f"[rank {_rank}] WARNING: no per-rank RNG in {_resume_ckpt_dir or 'checkpoint'}; "
                    "restored the saving rank's state. Data order will not match an uninterrupted run."
                )
        if not _rng_restored:
            print(f"[rank {_rank}] WARNING: checkpoint carries no RNG state; reseeding from config seed.")

    checkpoint = None  # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # type: ignore[assignment] # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])  # type: ignore[assignment]


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# Hugging Face style checkpoint management
def convert_nanogpt_to_hf_state_dict(model_state, model_args):
    """
    Convert nanoGPT state_dict to HuggingFace GPT2LMHeadModel compatible format.

    nanoGPT uses nn.Linear for attention, while HuggingFace uses Conv1D.
    Conv1D weights are transposed compared to Linear weights.
    """
    hf_state_dict = {}

    # Keys that need to be transposed (Conv1D in HF vs Linear in nanoGPT)
    transposed_keys = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for key, value in model_state.items():
        # Skip attention mask buffers
        if key.endswith(".attn.bias"):
            continue

        # Check if this key needs transposition
        needs_transpose = any(key.endswith(t) for t in transposed_keys)

        if needs_transpose:
            hf_state_dict[key] = value.t().contiguous()
        else:
            hf_state_dict[key] = value

    return hf_state_dict


def save_checkpoint_hf(
    model_state, optimizer_state, model_args, iter_num, val_loss, config, checkpoint_dir, early_stopping_counter=0, best_val_step=None
):
    """
    Save checkpoint in HuggingFace Transformers compatible format.

    This creates:
    - config.json: Model configuration (GPT2Config compatible)
    - pytorch_model.bin: Model weights (HuggingFace compatible state_dict)
    - training_state.bin: Training state (optimizer, iteration, etc.) for resuming
    - training_args.json: Training arguments as JSON
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Save HuggingFace compatible config.json
    hf_config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": model_args["vocab_size"],
        "n_positions": model_args["block_size"],
        "n_ctx": model_args["block_size"],
        "n_embd": model_args["n_embd"],
        "n_layer": model_args["n_layer"],
        "n_head": model_args["n_head"],
        "n_inner": model_args["n_embd"] * 4,  # GPT-2 uses 4x expansion
        "activation_function": "gelu_new",
        "resid_pdrop": model_args.get("dropout", 0.0),
        "embd_pdrop": model_args.get("dropout", 0.0),
        "attn_pdrop": model_args.get("dropout", 0.0),
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "use_cache": True,
        "bos_token_id": config.get("bos_token_id", 0),
        "eos_token_id": config.get("eos_token_id", config.get("eos_token", 0)),
        "transformers_version": "4.0.0",
        # Custom fields for our training
        "_name_or_path": "riken-gpt2",
        "_riken_model_args": model_args,
        "_riken_bias": model_args.get("bias", False),
    }
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    # 2. Save model weights in HuggingFace compatible format
    # Convert nanoGPT state_dict to HuggingFace format
    hf_state_dict = convert_nanogpt_to_hf_state_dict(model_state, model_args)
    torch.save(hf_state_dict, os.path.join(checkpoint_dir, "pytorch_model.bin"))

    # 3. Save training state separately (for resuming training)
    training_state = {
        "model": model_state,  # Original nanoGPT format for resume
        "optimizer": optimizer_state,
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": val_loss,
        "best_val_step": best_val_step,
        "early_stopping_counter": early_stopping_counter,
        "config": config,
        # Saving process only; ranks write their own rng_state_{rank}.pth
        # alongside this file. Kept here so single-process runs resume exactly
        # without depending on the per-rank files.
        "rng_state": capture_rng_state(),
    }
    torch.save(training_state, os.path.join(checkpoint_dir, "training_state.bin"))

    # 4. Save training args as JSON
    training_args = {
        "iteration": iter_num,
        "best_val_loss": float(val_loss) if val_loss is not None else None,
        "early_stopping_counter": early_stopping_counter,
        "learning_rate": config.get("learning_rate"),
        "batch_size": config.get("batch_size"),
        "block_size": config.get("block_size"),
        "model_args": model_args,
    }
    with open(os.path.join(checkpoint_dir, "training_args.json"), "w") as f:
        json.dump(training_args, f, indent=2)

    print(f"Checkpoint saved to {checkpoint_dir} (HuggingFace compatible format)")


def cleanup_old_checkpoints(base_dir, max_checkpoints, protect_step=None):
    """Remove old checkpoints, keeping only the most recent max_checkpoints.

    ``protect_step`` (optional): step number of a checkpoint that must NOT be
    deleted even if it falls outside the max_checkpoints window (used to preserve
    the best-val checkpoint for downstream evaluation; charter § 1.1 comparability
    rule requires subset comparison at best-val ckpt).
    """
    if max_checkpoints is None:
        return

    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return

    # Extract step numbers and sort
    checkpoints = []
    for ckpt_dir in checkpoint_dirs:
        try:
            step = int(os.path.basename(ckpt_dir).split("-")[1])
            checkpoints.append((step, ckpt_dir))
        except (ValueError, IndexError):
            continue

    checkpoints.sort(reverse=True)  # Sort by step, newest first

    # Remove old checkpoints, but PROTECT the best-val step (if provided)
    for _step, ckpt_dir in checkpoints[max_checkpoints:]:
        if protect_step is not None and _step == protect_step:
            # Best-val ckpt: skip deletion to preserve for downstream evaluation
            continue
        print(f"Removing old checkpoint: {ckpt_dir}")
        shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    # training loop
    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Reset each iteration; the master-only early-stop check below may set
        # it, then it is broadcast to every rank so they all break together.
        should_stop = False

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            with open(logging_file, "a") as f:
                f.write(f"{iter_num}, {losses['train']:.4f}, {losses['val']:.4f}\n")

            if writer is not None:
                writer.add_scalar("Val Loss", losses["val"], iter_num)
                writer.flush()

            # Log to wandb
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/train_loss": losses["train"],
                        "eval/val_loss": losses["val"],
                        "eval/iter": iter_num,
                        "eval/best_val_loss": best_val_loss,
                    },
                    step=iter_num,
                )

            # Track best validation loss for early stopping
            is_best_model = False
            if losses["val"] < best_val_loss:
                # Validation loss improved
                best_val_loss = losses["val"]
                best_val_step = iter_num  # protect this ckpt from FIFO eviction (charter § 1.1)
                early_stopping_counter = 0
                is_best_model = True
            else:
                # Validation loss did not improve
                early_stopping_counter += 1
                if early_stopping and early_stopping_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {early_stopping_counter} evaluations without improvement.")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    # Defer the break: all ranks must agree (broadcast below), or
                    # rank 0 leaves and the others hang at the next NCCL collective.
                    should_stop = True

            # Checkpoint saving logic (independent of best model tracking).
            # These reasons are additive, not exclusive: setting
            # save_checkpoint_steps asks for periodic checkpoints *in addition
            # to* the best-val one, so it must not suppress the best-val save.
            # (When these were chained with elif, enabling save_checkpoint_steps
            # silently dropped every improvement that did not land on a
            # multiple of the interval, defeating the best-ckpt protection that
            # subset comparison depends on.)
            should_save_checkpoint = (
                # Save at every eval_interval
                always_save_checkpoint
                # Save at specific step intervals (periodic, for resume/chaining)
                or (save_checkpoint_steps is not None and iter_num % save_checkpoint_steps == 0)
                # Save when validation improves
                or is_best_model
            )

            if should_save_checkpoint and iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),  # type: ignore[union-attr]
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "early_stopping_counter": early_stopping_counter,
                    "config": config,
                    "rng_state": capture_rng_state(),
                }

                # Save in Hugging Face format
                if save_hf_checkpoints:
                    checkpoint_dir = os.path.join(out_dir, f"checkpoint-{iter_num}")
                    save_checkpoint_hf(
                        raw_model.state_dict(),  # type: ignore[union-attr]
                        optimizer.state_dict(),
                        model_args,
                        iter_num,
                        best_val_loss,
                        config,
                        checkpoint_dir,
                        early_stopping_counter,
                        best_val_step=best_val_step,
                    )

                    # Log checkpoint to wandb as artifact BEFORE cleanup
                    if wandb_run is not None and wandb_log_model:
                        artifact = wandb.Artifact(  # type: ignore[attr-defined]
                            name=f"model-{wandb_run.id}",
                            type="model",
                            description=f"Model checkpoint at step {iter_num}",
                            metadata={
                                "iter": iter_num,
                                "train_loss": float(losses["train"]),
                                "val_loss": float(losses["val"]),
                                "best_val_loss": float(best_val_loss),
                            },
                        )
                        artifact.add_dir(checkpoint_dir)
                        wandb_run.log_artifact(artifact)

                    # Cleanup old checkpoints AFTER wandb upload.
                    # Protect best-val step from FIFO eviction (charter § 1.1
                    # comparability rule: subsets are compared on the best-val ckpt).
                    cleanup_old_checkpoints(out_dir, max_checkpoints, protect_step=best_val_step)

                # Also save legacy ckpt.pt for backward compatibility (or best model)
                if keep_legacy_ckpt or is_best_model:
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

        # Per-rank RNG for the periodic checkpoints a chained run resumes from.
        # Every rank draws its own batches from its own np.random stream, so
        # rank 0's state alone cannot restore the others.
        #
        # The condition is recomputed here rather than reusing the decision
        # above because that one lives under `master_process` and also depends
        # on is_best_model, which only rank 0 knows. Periodic saves are a pure
        # function of iter_num, so every rank agrees without communicating --
        # deliberately avoiding a new collective on the eval path, which is
        # where the NCCL watchdog stalls noted above have shown up.
        # Mirrors the schedule-driven arms of the decision above -- both
        # always_save_checkpoint and save_checkpoint_steps are configuration,
        # so they hold identically on every rank. The is_best_model arm is
        # excluded because only rank 0 can evaluate it; best-val checkpoints
        # are what comparison reads, not what chaining resumes from.
        if (
            save_hf_checkpoints
            and iter_num > 0
            and iter_num % eval_interval == 0
            and (
                always_save_checkpoint
                or (save_checkpoint_steps is not None and iter_num % save_checkpoint_steps == 0)
            )
        ):
            _ckpt_dir = os.path.join(out_dir, f"checkpoint-{iter_num}")
            os.makedirs(_ckpt_dir, exist_ok=True)  # rank 0 may not have created it yet
            torch.save(capture_rng_state(), rank_rng_path(_ckpt_dir, ddp_rank if ddp else 0))

        # Early stopping is decided only on rank 0 (evaluation runs there).
        # Broadcast the decision so every rank breaks on the same iteration;
        # otherwise non-master ranks block forever on the next collective
        # (seen as a NCCL watchdog timeout that killed multi-GPU runs at the
        # stop point). Mirrors how the max_iters termination breaks all ranks.
        if early_stopping and iter_num % eval_interval == 0:
            if ddp:
                _stop = torch.tensor(1 if should_stop else 0, device=device)
                torch.distributed.broadcast(_stop, src=0)
                should_stop = bool(_stop.item())
            if should_stop:
                break

        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1  # type: ignore[assignment]
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt, flops_promised=mfu_peak_tflops * 1e12)  # type: ignore[union-attr,operator]
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
            if writer is not None:
                writer.add_scalar("Loss", lossf, iter_num)
                writer.add_scalar("Learning Rate", lr, iter_num)
                writer.flush()

            # Log training metrics to wandb
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": lossf,
                        "train/lr": lr,
                        "train/mfu": running_mfu,
                        "train/iter": iter_num,
                        "train/dt_ms": dt * 1000,
                    },
                    step=iter_num,
                )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()

    if ddp:
        destroy_process_group()
