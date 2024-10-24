import os
import numpy as np
from argparse import ArgumentParser
from molecule_related_nl.utils.general import read_dataset
from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer
import random
from tqdm import tqdm
import torch


def generate_sample_stacks(data, stack_size):
    stacks = []
    current_stack_size = stack_size
    current_prepared_sample = []

    for i in tqdm(range(len(data))):
        sample = data[i]["input_ids"] + data[i]["output_ids"]
        current_stack_size -= len(sample)

        if current_stack_size < 0:
            current_prepared_sample.extend(sample[:current_stack_size])
            stacks.append(current_prepared_sample)
            current_prepared_sample = sample[current_stack_size:]

            if len(current_prepared_sample) > stack_size:
                current_prepared_sample = current_prepared_sample[stack_size:]

            current_stack_size = stack_size - len(current_prepared_sample)

        elif current_stack_size >= 0:
            current_prepared_sample.extend(sample)


    return stacks


parser = ArgumentParser()
parser.add_argument("tokenized_file_path", type=str, help="Path to the tokenized file")
parser.add_argument("-train-s", "--training-set-subset-len", type=str, help="Length of the training set subset. If < 1 taken as fracation of full data. If > 1 taken as number of samples.", default=10_000)
parser.add_argument("-test-s","--test-set-subset-len", type=str, help="Length of the training set subset. If training-set-subset-len < 1, then this parameter is overriden. If > 1 taken as number of samples.", default=1_000)
args = parser.parse_args()

data = read_dataset(args.tokenized_file_path)

if args.training_set_subset_len <= 1:
    train_num_samples = int(len(data["train"])*args.training_set_subset_len)
    test_num_samples = int(len(data["test"])*args.training_set_subset_len)
else:
    train_num_samples = min(len(data["train"]), args.training_set_subset_len)
    test_num_samples = min(len(data["test"]), args.training_set_subset_len)

indices = [i for i in range(len(data["train"]))]
train_idx = random.sample(indices, train_num_samples)

indices = [i for i in range(len(data["test"])) if i not in train_idx]
test_idx = random.sample(indices, test_num_samples)

train_data = data["train"].select(train_idx)
test_data = data["test"].select(test_idx)

stacked_train_data = generate_sample_stacks(train_data, 1024)
stacked_test_data = generate_sample_stacks(test_data, 1024)

# export to bin files
torch.save(torch.tensor(stacked_train_data), os.path.join(os.path.dirname(__file__), 'train.pt'))
torch.save(torch.tensor(stacked_test_data), os.path.join(os.path.dirname(__file__), 'val.pt'))
