"""
Helper script for GPT2 checkpoint testing
Automatically discover and test checkpoints for each domain.

# List checkpoints
python gpt2/test_helper.py --search_dir=runs_train_bert_*/checkpoints --list_only

# test a specific checkpoint
python gpt2/test_helper.py --checkpoint_path=path/to/ckpt.pt

# Automatically test all checkpoints
python gpt2/test_helper.py --search_dir=runs_* --auto_run

"""

import argparse
import glob
import json
import os
from pathlib import Path


def find_checkpoint_files(search_dir):
    """Search checkpoint file"""
    checkpoint_patterns = [
        "**/ckpt.pt",
        "**/checkpoint.pt",
        "**/pytorch_model.bin",
        "**/model.safetensors",
        "**/*checkpoint*.pt",
    ]

    checkpoints = []
    for pattern in checkpoint_patterns:
        found = glob.glob(os.path.join(search_dir, pattern), recursive=True)
        checkpoints.extend(found)

    return list(set(checkpoints))  # remove duplicates


def get_domain_info():
    """Returns information for each domain"""
    import os
    import sys

    from molcrawl.core.paths import COMPOUNDS_DATASET_DIR, MOLECULE_NAT_LANG_DATASET_DIR

    return {
        "compounds": {
            "vocab_path": "assets/molecules/vocab.txt",
            "dataset_dir": COMPOUNDS_DATASET_DIR,
        },
        "molecule_nat_lang": {"vocab_path": None, "dataset_dir": MOLECULE_NAT_LANG_DATASET_DIR},
        "genome_sequence": {
            "vocab_path": None,  # SentencePiece model path required
            "dataset_dir": "outputs/genome_sequence/training_ready_hf_dataset",
        },
        "protein_sequence": {
            "vocab_path": None,
            "dataset_dir": "outputs/protein_sequence/training_ready_hf_dataset",
        },
        "rna": {
            "vocab_path": None,
            "dataset_dir": "outputs/rna/training_ready_hf_dataset",
        },
    }


def detect_domain_from_path(checkpoint_path):
    """Guess domain from checkpoint path"""
    path_lower = checkpoint_path.lower()

    if "compound" in path_lower:
        return "compounds"
    elif "molecule" in path_lower and "nl" in path_lower:
        return "molecule_nat_lang"
    elif "genome" in path_lower:
        return "genome_sequence"
    elif "protein" in path_lower:
        return "protein_sequence"
    elif "rna" in path_lower:
        return "rna"
    else:
        return None


def create_test_command(checkpoint_path, domain=None, output_dir=None, max_samples=500):
    """Generate test command"""
    domain_info = get_domain_info()

    # guess domain
    if domain is None:
        domain = detect_domain_from_path(checkpoint_path)

    if domain is None:
        print(f"Warning: Could not infer domain from {checkpoint_path}")
        domain = "unknown"

    # set output directory
    if output_dir is None:
        checkpoint_name = Path(checkpoint_path).parent.name
        output_dir = f"test_results_{domain}_{checkpoint_name}"

    # Basic command
    cmd = [
        "python",
        "gpt2/test_checkpoint.py",
        f"--checkpoint_path={checkpoint_path}",
        f"--output_dir={output_dir}",
        f"--max_test_samples={max_samples}",
        "--convert_to_hf",
    ]

    # Domain-specific settings
    if domain in domain_info:
        cmd.append(f"--domain={domain}")

        vocab_path = domain_info[domain]["vocab_path"]
        if vocab_path and os.path.exists(vocab_path):
            cmd.append(f"--vocab_path={vocab_path}")

        dataset_dir = domain_info[domain]["dataset_dir"]
        if os.path.exists(dataset_dir):
            dataset_params = {"dataset_dir": dataset_dir}
            cmd.append(f"--test_dataset_params={json.dumps(dataset_params)}")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="GPT2 checkpoint test helper")
    parser.add_argument("--search_dir", default=".", help="Directory to search for checkpoints")
    parser.add_argument("--checkpoint_path", help="Specific checkpoint path")
    parser.add_argument(
        "--domain",
        choices=["compounds", "molecule_nat_lang", "genome_sequence", "protein_sequence", "rna"],
        help="Forcibly specify domain",
    )
    parser.add_argument("--output_dir", help="output directory")
    parser.add_argument("--max_samples", type=int, default=500, help="Number of test samples")
    parser.add_argument("--auto_run", action="store_true", help="Run automatically")
    parser.add_argument("--list_only", action="store_true", help="Only list checkpoints")

    args = parser.parse_args()

    if args.checkpoint_path:
        # test a specific checkpoint
        checkpoints = [args.checkpoint_path]
    else:
        # Search for checkpoints
        print(f"Searching for checkpoint: {args.search_dir}")
        checkpoints = find_checkpoint_files(args.search_dir)

    if not checkpoints:
        print("Checkpoint not found.")
        return

    print(f"\nCheckpoints discovered: {len(checkpoints)}")
    for i, cp in enumerate(checkpoints, 1):
        domain = detect_domain_from_path(cp) or "unknown"
        size_mb = os.path.getsize(cp) / 1024 / 1024 if os.path.exists(cp) else 0
        print(f"{i:2d}. {cp} [{domain}] ({size_mb:.1f} MB)")

    if args.list_only:
        return

    # Generate test commands for each checkpoint
    for checkpoint in checkpoints:
        print(f"\n{'=' * 60}")
        print(f"Checkpoint: {checkpoint}")

        domain = args.domain or detect_domain_from_path(checkpoint)
        if domain:
            print(f"Detection domain: {domain}")

        cmd = create_test_command(
            checkpoint,
            domain=args.domain,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )

        print("Run command:")
        print(" ".join(cmd))

        if args.auto_run:
            print("\nRunning...")
            import subprocess

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✓ Test successful")
                else:
                    print(f"✗ Test failed: {result.stderr}")
            except Exception as e:
                print(f"✗ Execution error: {e}")
        else:
            print("\nPlease execute the above command to start the test.")


def create_test_configs():
    """Create a test configuration file for each domain"""
    configs_dir = Path("gpt2/test_configs")
    configs_dir.mkdir(exist_ok=True)

    domain_info = get_domain_info()

    for domain, info in domain_info.items():
        config_content = f"""# {domain.upper()} GPT2 test configuration for domain

# Basic settings
domain = "{domain}"
max_test_samples = 1000
convert_to_hf = True

# datasetsetting
"""

        if info["dataset_dir"]:
            config_content += f"""dataset_params = {{
    "dataset_dir": "{info["dataset_dir"]}"
}}
"""

        if info["vocab_path"]:
            config_content += f"""
# vocabulary file
vocab_path = "{info["vocab_path"]}"
"""

        config_content += f"""
# Output settings
output_dir = "test_results_{domain}"

# device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
"""

        config_file = configs_dir / f"{domain}_test_config.py"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)

        print(f"Create configuration file: {config_file}")


if __name__ == "__main__":
    # First create a configuration file
    print("Creating test configuration file...")
    create_test_configs()
    print()

    # Main processing
    main()
