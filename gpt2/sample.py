"""
Sample from a trained model
"""

import os
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

from core.dataset import PreparedDataset

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def extract_sublist(lst, start_token, end_sequence):

    start = (lst == start_token).nonzero(as_tuple=True)[0][0].item()
    
    end_seq_element = end_sequence[0]
    possible_end_seq_starts = (lst == end_seq_element).nonzero(as_tuple=True)[0]

    for i in possible_end_seq_starts:
        if i < start:
            continue

        if list(lst[i:i+len(end_sequence)]) == end_sequence:
            end = i + len(end_sequence)
            return lst[start:end]
        
    return None

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


training_data = PreparedDataset(**dataset_params, split="train")
test_data = PreparedDataset(**dataset_params, split="test")

ix = torch.randint(len(test_data), (10,))
inputs = torch.stack([test_data[int(i)] for i in ix])

if device_type == "cuda":
    inputs = inputs.pin_memory().to(device, non_blocking=True)
else:
    inputs = inputs.to(device)

# run generation
with torch.no_grad():
    with ctx:
        for x in inputs:
            if start_instruction is not None:
                prompt = extract_sublist(x, start_instruction, end_instruction)
                if prompt is None:
                    prompt = x[:256]
            y = model.generate(prompt.unsqueeze(0), max_new_tokens, temperature=temperature, top_k=top_k)

            response = tokenizer.decode(y[0, len(prompt):].tolist())

            if eos_token is not None:
                response = response.split(tokenizer.decode([eos_token]))[0]

            print(f"{bcolors.WARNING}{tokenizer.decode(prompt.tolist())}{bcolors.ENDC}{bcolors.OKBLUE}{response}{bcolors.ENDC}")
            print("---------------")
