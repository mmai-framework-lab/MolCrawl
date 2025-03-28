"""
Sample from a trained model
"""

import os
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from tqdm import tqdm

from core.dataset import PreparedDataset


# Special Tokens
start_instruction = None
end_instruction = None
eos_token = None

dataset_params = {}
dataset = ""
# -----------------------------------------------------------------------------
init_from = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = "out"  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open("gpt2/configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

try:
    eval_batch_size
except NameError:
    eval_batch_size = batch_size

# data = PreparedDataset(**dataset_params, split="test")
data = PreparedDataset(**dataset_params, split="valid")

def get_batch(eval_batch_size):
    current = 0

    while current + eval_batch_size < len(data):
        batch = data[torch.arange(current, current + eval_batch_size)]
        x = batch[:, :-1]
        y = batch[:, 1:]
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        yield x, y
        current += eval_batch_size

@torch.no_grad()
def estimate_loss():
    print("Estimating validation loss...")
    losses = []
    with tqdm(total=(len(data)//eval_batch_size)) as pbar:

        for x, y in get_batch(eval_batch_size):
            with ctx:
                logits, loss = model(x, y)
            losses.append(loss.item())
            pbar.update(1)

    return torch.tensor(losses).mean().item()

loss = estimate_loss()
print(f"File: {out_dir}, val loss {loss:.4f}")
