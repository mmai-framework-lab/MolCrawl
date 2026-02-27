#!/usr/bin/env python3
"""
ChemBERTa-2 Configuration File Loader

Loads configuration from a Python file and allows command-line overrides.
"""

import argparse
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_default_config = os.path.join(_this_dir, "configs", "compounds.py")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="ChemBERTa-2 Training Configuration")
parser.add_argument("--config", type=str, default=_default_config, help="Path to config file")
parser.add_argument("--model_size", type=str, choices=["small", "medium", "large"], help="Model size")
parser.add_argument("--use_wandb", type=str, choices=["True", "False"], help="Enable Weights & Biases logging")
parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
parser.add_argument("--max_steps", type=int, help="Maximum training steps")
parser.add_argument("--learning_rate", type=float, help="Learning rate")
parser.add_argument("--batch_size", type=int, help="Batch size per device")
parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")

args, unknown = parser.parse_known_args()

# Load config file only when executed via exec() from main.py (__name__ == "__main__"),
# not when pdoc or other tools import this module directly as src.chemberta2.configurator.
if os.path.exists(args.config) and __name__ == "__main__":
    print(f"📝 Loading config from: {args.config}")
    with open(args.config, "r") as f:
        exec(f.read())

if __name__ == "__main__":
    # Override with command-line arguments
    if args.model_size is not None:
        model_size = args.model_size
        print(f"   ⚙️  Overriding model_size: {model_size}")

    if args.use_wandb is not None:
        use_wandb = args.use_wandb == "True"
        print(f"   ⚙️  Overriding use_wandb: {use_wandb}")

    if args.wandb_project is not None:
        wandb_project = args.wandb_project
        print(f"   ⚙️  Overriding wandb_project: {wandb_project}")

    if args.max_steps is not None:
        max_steps = args.max_steps
        print(f"   ⚙️  Overriding max_steps: {max_steps}")

    if args.learning_rate is not None:
        learning_rate = args.learning_rate
        print(f"   ⚙️  Overriding learning_rate: {learning_rate}")

    if args.batch_size is not None:
        batch_size = args.batch_size
        print(f"   ⚙️  Overriding batch_size: {batch_size}")

    if args.gradient_accumulation_steps is not None:
        gradient_accumulation_steps = args.gradient_accumulation_steps
        print(f"   ⚙️  Overriding gradient_accumulation_steps: {gradient_accumulation_steps}")
