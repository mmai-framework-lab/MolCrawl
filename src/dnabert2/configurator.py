"""
Configurator for DNABERT-2 training
Handles command line arguments and config file loading
Based on bert/configurator.py
"""

import argparse
import os
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description="DNABERT-2 Training Configuration")
parser.add_argument("config", type=str, help="Path to the config file")
parser.add_argument("--use_wandb", type=str, default=None, help="Enable wandb logging (True/False)")
parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")
parser.add_argument("--model_size", type=str, default=None, help="Model size (small/medium/large)")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every N steps")

args = parser.parse_args()

# Load config file
config_path = args.config
if not os.path.exists(config_path):
    print(f"❌ Config file not found: {config_path}")
    sys.exit(1)

print(f"📝 Loading config from: {config_path}")
exec(open(config_path).read())

# Override with command line arguments if provided
if args.use_wandb is not None:
    use_wandb = args.use_wandb.lower() in ("true", "1", "yes")
    print(f"   Overriding use_wandb: {use_wandb}")

if args.wandb_project is not None:
    wandb_project = args.wandb_project
    print(f"   Overriding wandb_project: {wandb_project}")

if args.wandb_run_name is not None:
    wandb_run_name = args.wandb_run_name
    print(f"   Overriding wandb_run_name: {wandb_run_name}")

if args.wandb_entity is not None:
    wandb_entity = args.wandb_entity
    print(f"   Overriding wandb_entity: {wandb_entity}")

if args.model_size is not None:
    model_size = args.model_size
    print(f"   Overriding model_size: {model_size}")

if args.max_steps is not None:
    max_steps = args.max_steps
    print(f"   Overriding max_steps: {max_steps}")

if args.learning_rate is not None:
    learning_rate = args.learning_rate
    print(f"   Overriding learning_rate: {learning_rate}")

if args.batch_size is not None:
    batch_size = args.batch_size
    print(f"   Overriding batch_size: {batch_size}")

if args.save_steps is not None:
    save_steps = args.save_steps
    print(f"   Overriding save_steps: {save_steps}")

print("✅ Configuration loaded")
