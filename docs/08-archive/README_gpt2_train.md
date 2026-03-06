# GPT Training Script Overview (Archived)

## Purpose

This document summarizes a comprehensive GPT-2 training system intended for large-scale language-model pretraining.

## Core Capabilities

- Supports both **single-GPU** and **distributed (DDP)** training
- Supports **multi-node distributed training**
- Supports **checkpoint save/resume**
- Supports **TensorBoard logging and visualization**
- Supports **mixed-precision training** for speed and memory efficiency

## Execution Modes

```bash
# 1) Single-GPU training
python train.py --batch_size=32 --compile=False

# 2) Single-node, 4-GPU distributed training
torchrun --standalone --nproc_per_node=4 train.py

# 3) Multi-node distributed training (2 nodes, 8 GPUs each)
# Master node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=123.456.123.456 --master_port=1234 train.py
# Worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=123.456.123.456 --master_port=1234 train.py
```

## Example Default Hyperparameters

```python
# Model architecture
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024

# Training
batch_size = 12
gradient_accumulation_steps = 40
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1

# LR schedule
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
```

## Data Pipeline Sketch

```python
training_data = PreparedDataset(**dataset_params, split="train")
val_data = PreparedDataset(**dataset_params, split="valid")

def get_batch(split):
    data = training_data if split == "train" else val_data
    ix = np.random.randint(0, len(data), batch_size).tolist()
    batch = torch.stack([data[i] for i in ix])

    x = batch[:, :-1]  # input
    y = batch[:, 1:]   # next-token target
    return x.to(device), y.to(device)
```

## Training Loop Skeleton

```python
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    iter_num += 1
    if iter_num > max_iters:
        break
```

## DDP Initialization Sketch

```python
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)

    master_process = ddp_rank == 0
    gradient_accumulation_steps //= ddp_world_size
    model = DDP(model, device_ids=[ddp_local_rank])
```

## Logging and Monitoring

```python
# CSV logging
with open(logging_file, "a") as f:
    f.write(f"{iter_num}, {losses['train']:.4f}, {losses['val']:.4f}\\n")

# TensorBoard
if writer is not None:
    writer.add_scalar("Loss", lossf, iter_num)
    writer.add_scalar("Learning Rate", lr, iter_num)
    writer.add_scalar("Val Loss", losses['val'], iter_num)
```

## Optimization Techniques

- Mixed precision (FP16/BF16)
- Gradient accumulation
- Gradient clipping
- Cosine LR decay
- Optional PyTorch compile path

## Applicability

This training design can be adapted to:

- Genome sequence models
- Compound generation models
- Protein sequence models
- General NLP models
