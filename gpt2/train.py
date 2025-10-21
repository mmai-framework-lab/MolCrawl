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

import os
import time
import math
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from core.dataset import PreparedDataset
from datasets import load_from_disk, Dataset
import json
import pandas as pd
from pathlib import Path
import pyarrow as pa

class RNADataset:
    """RNA Transcriptome Dataset"""
    
    def __init__(self, data_dir, split="train", vocab_file=None, test_size=0.1):
        self.data_dir = data_dir
        self.split = split
        self.test_size = test_size
        
        # Load vocabulary
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
            self.vocab_size = len(self.vocab)
        else:
            # Default RNA vocabulary
            self.vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
            self.vocab_size = 3
        
        # Load dataset - direct arrow file reading to bypass metadata issues
        print(f"📂 Attempting to load data from {data_dir}")
        
        try:
            data_path = Path(data_dir)
            arrow_files = sorted(list(data_path.glob("*.arrow")))
            
            if arrow_files:
                print(f"📁 Found {len(arrow_files)} arrow files: {[f.name for f in arrow_files]}")
                
                all_batches = []
                for arrow_file in arrow_files:
                    print(f"📖 Reading {arrow_file.name}...")
                    try:
                        # Try as memory mapped stream first
                        with pa.memory_map(str(arrow_file)) as mmap:
                            with pa.ipc.open_stream(mmap) as reader:
                                table = reader.read_all()
                                print(f"� Read table via stream: {len(table)} rows")
                                all_batches.append(table)
                    except Exception:
                        try:
                            # Fallback to RecordBatch file
                            with pa.memory_map(str(arrow_file)) as mmap:
                                with pa.ipc.open_file(mmap) as reader:
                                    table = reader.read_all()
                                    print(f"� Read table via file: {len(table)} rows")
                                    all_batches.append(table)
                        except Exception as e:
                            print(f"❌ Failed to read {arrow_file.name}: {e}")
                            continue
                
                if all_batches:
                    # Combine all tables
                    combined_table = pa.concat_tables(all_batches)
                    print(f"📊 Combined {len(all_batches)} tables: {len(combined_table)} total rows")
                    
                    # Convert PyArrow table to pandas DataFrame, then to HuggingFace Dataset
                    df = combined_table.to_pandas()
                    print(f"📋 Converted to pandas DataFrame: {len(df)} rows")
                    
                    # Convert numpy arrays to lists for HuggingFace compatibility
                    if 'token' in df.columns:
                        df['token'] = df['token'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
                    
                    # Create dataset from pandas DataFrame (bypasses metadata issues)
                    self.dataset = Dataset.from_pandas(df)
                    print(f"✅ Created HuggingFace Dataset from pandas DataFrame")
                    print(f"🔍 Dataset columns: {self.dataset.column_names}")
                else:
                    raise ValueError("No arrow files could be read successfully")
            else:
                raise FileNotFoundError(f"No .arrow files found in {data_dir}")
                
        except Exception as e:
            print(f"❌ Arrow loading failed: {e}")
            # Fallback to other methods
            try:
                print("🔄 Trying HuggingFace format as fallback...")
                self.dataset = load_from_disk(data_dir)
                print(f"✅ Loaded HuggingFace dataset from {data_dir}")
            except Exception as e2:
                print(f"❌ All loading methods failed: {e2}")
                raise FileNotFoundError(f"Could not load data from {data_dir}")
        
        # Split into train/valid if needed
        if hasattr(self.dataset, 'keys') and isinstance(self.dataset, dict) and 'train' in self.dataset:
            # Already has splits
            if split == "train":
                self.data = self.dataset['train']
            elif split == "valid" or split == "val":
                self.data = self.dataset.get('valid', self.dataset.get('test', self.dataset['train']))
        else:
            # Single dataset, need to split
            total_size = len(self.dataset)
            if split == "train":
                self.data = self.dataset.select(range(int(total_size * (1 - self.test_size))))
            else:  # valid
                self.data = self.dataset.select(range(int(total_size * (1 - self.test_size)), total_size))
        
        print(f"Loaded {len(self.data)} samples for {split}")
        
        # Sample a few examples to understand data structure
        if len(self.data) > 0:
            sample = self.data[0]
            print(f"Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, (list, str)):
                    print(f"  {key}: {type(value)} of length {len(value)}")
                else:
                    print(f"  {key}: {type(value)} = {value}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # RNA data has 'token' column with numpy arrays
        tokens = None
        
        # Try 'token' column first (RNA data format)
        if 'token' in item and item['token'] is not None:
            tokens = item['token']
        else:
            # Try other possible token column names
            for key in ['input_ids', 'tokens', 'token_ids', 'tokenized']:
                if key in item and item[key] is not None:
                    tokens = item[key]
                    break
        
        if tokens is None:
            # If no tokens, try to find text and tokenize it
            text = None
            for key in ['text', 'sequence', 'input_text']:
                if key in item and item[key] is not None:
                    text = item[key]
                    break
            
            if text is not None:
                # Simple tokenization (this is a fallback)
                tokens = [self.vocab.get(char, self.vocab.get('<unk>', 1)) for char in str(text)]
            else:
                # Last resort: use all numeric values as tokens
                numeric_values = [v for v in item.values() if isinstance(v, (int, list))]
                if numeric_values:
                    tokens = numeric_values[0] if isinstance(numeric_values[0], list) else [numeric_values[0]]
                else:
                    tokens = [0]  # padding token
        
        # Handle numpy array or list
        if hasattr(tokens, 'tolist'):
            # Convert numpy array to list
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            tokens = list(tokens)
        
        # Convert to integers if needed
        try:
            tokens = [int(t) for t in tokens]
        except (ValueError, TypeError):
            tokens = [self.vocab.get('<unk>', 1) for _ in tokens]
        
        return torch.tensor(tokens, dtype=torch.long)

dataset_params = {}
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

tensorboard = False  # log training metrics to tensorboard
tensorboard_dir = "runs"

out_dir = "out-gpt2"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
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
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
# Handle configurator path
configurator_path = "gpt2/configurator.py" if os.path.exists("gpt2/configurator.py") else "configurator.py"
exec(open(configurator_path).read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# create folder if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# create empty csv file for logging
logging_file = os.path.join(out_dir, "logging.csv")

with open(logging_file, "w") as f:
    f.write("iter, train_loss, val_loss\n")

writer = None
if tensorboard:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(tensorboard_dir)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
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
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# RNA data loader
rna_data_dir = "path-to-rna-parquet" # TODO
rna_vocab_file = "path-to-rna-vocab" # TODO

# Use RNADataset if dataset is "rna", otherwise use PreparedDataset
if dataset == "rna":
    training_data = RNADataset(rna_data_dir, split="train", vocab_file=rna_vocab_file, test_size=0.1)
    test_data = RNADataset(rna_data_dir, split="valid", vocab_file=rna_vocab_file, test_size=0.1)
    # Set vocab size from the RNA dataset
    meta_vocab_size = training_data.vocab_size
else:
    print (f"Loading dataset: {dataset_params}")
    training_data = PreparedDataset(**dataset_params, split="train")
    test_data = PreparedDataset(**dataset_params, split="valid")

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
    y = batch[:, 1:].long()   # Ensure long type for embedding
    
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

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
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout
)  # start with model_args from command line

if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
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
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


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

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        with open(logging_file, "a") as f:
            f.write(f"{iter_num}, {losses['train']:.4f}, {losses['val']:.4f}\n")

        if writer is not None:
            writer.add_scalar("Val Loss", losses['val'], iter_num)
            writer.flush()

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
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
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
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
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if writer is not None:
            writer.add_scalar("Loss", lossf, iter_num)
            writer.add_scalar("Learning Rate", lr, iter_num)
            writer.flush()
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
