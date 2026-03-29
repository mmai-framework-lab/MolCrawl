"""
Comprehensive testing and validation script for GPT2 checkpoints

This script provides the following functionality:
1. Convert GPT2 checkpoints to Hugging Face format
2. Verification of correct answer rate using test data
3. Perplexity calculation
4. Text generation quality evaluation
5. Model performance statistics
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from molcrawl.core.dataset import PreparedDataset
from molcrawl.gpt2.model import GPT, GPTConfig

# Add the project's src directory to the path
# import GPT2 model class


def load_gpt2_checkpoint(checkpoint_path, device="cuda"):
    """
    Load custom GPT2 checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # get model settings
        model_args = checkpoint["model_args"]
        print(f"Model settings: {model_args}")

        # create GPTConfig
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # load state dictionary
        state_dict = checkpoint["model"]

        # remove unnecessary prefixes
        unwanted_prefix = "_orig_mod."
        for k, _v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print("✓ Checkpoint read successfully")

        # Additional Information
        iter_num = checkpoint.get("iter_num", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        config = checkpoint.get("config", {})

        return model, model_args, iter_num, best_val_loss, config

    except Exception as e:
        print(f"✗ Error occurred while loading checkpoint: {e}")
        return None, None, None, None, None


def convert_to_hf_format(model, model_args, output_dir):
    """
    Convert custom GPT models to Hugging Face format
    """
    print(f"Converting to Hugging Face format: {output_dir}")

    try:
        # Create HF GPT2Config
        hf_config = GPT2Config(
            vocab_size=model_args["vocab_size"],
            n_positions=model_args["block_size"],
            n_embd=model_args["n_embd"],
            n_layer=model_args["n_layer"],
            n_head=model_args["n_head"],
            use_cache=True,
            bos_token_id=0,
            eos_token_id=0,
        )

        # Create HF GPT2LMHeadModel
        hf_model = GPT2LMHeadModel(hf_config)

        # map weights
        state_dict_mapping = {}

        # Mapping Transformer block weights
        for i in range(model_args["n_layer"]):
            # Attention weights
            state_dict_mapping[f"transformer.h.{i}.attn.c_attn.weight"] = f"transformer.h.{i}.attn.c_attn.weight"
            state_dict_mapping[f"transformer.h.{i}.attn.c_proj.weight"] = f"transformer.h.{i}.attn.c_proj.weight"

            # LayerNorm weights
            state_dict_mapping[f"transformer.h.{i}.ln_1.weight"] = f"transformer.h.{i}.ln_1.weight"
            state_dict_mapping[f"transformer.h.{i}.ln_2.weight"] = f"transformer.h.{i}.ln_2.weight"

            # MLP weights
            state_dict_mapping[f"transformer.h.{i}.mlp.c_fc.weight"] = f"transformer.h.{i}.mlp.c_fc.weight"
            state_dict_mapping[f"transformer.h.{i}.mlp.c_proj.weight"] = f"transformer.h.{i}.mlp.c_proj.weight"

            # Bias terms (if exists)
            if model_args.get("bias", False):
                state_dict_mapping[f"transformer.h.{i}.attn.c_attn.bias"] = f"transformer.h.{i}.attn.c_attn.bias"
                state_dict_mapping[f"transformer.h.{i}.attn.c_proj.bias"] = f"transformer.h.{i}.attn.c_proj.bias"
                state_dict_mapping[f"transformer.h.{i}.ln_1.bias"] = f"transformer.h.{i}.ln_1.bias"
                state_dict_mapping[f"transformer.h.{i}.ln_2.bias"] = f"transformer.h.{i}.ln_2.bias"
                state_dict_mapping[f"transformer.h.{i}.mlp.c_fc.bias"] = f"transformer.h.{i}.mlp.c_fc.bias"
                state_dict_mapping[f"transformer.h.{i}.mlp.c_proj.bias"] = f"transformer.h.{i}.mlp.c_proj.bias"

        # Other important weights
        state_dict_mapping["transformer.wte.weight"] = "transformer.wte.weight"  # Token embeddings
        state_dict_mapping["transformer.wpe.weight"] = "transformer.wpe.weight"  # Position embeddings
        state_dict_mapping["transformer.ln_f.weight"] = "transformer.ln_f.weight"  # Final LayerNorm
        state_dict_mapping["lm_head.weight"] = "lm_head.weight"  # Language model head

        if model_args.get("bias", False):
            state_dict_mapping["transformer.ln_f.bias"] = "transformer.ln_f.bias"

        # copy weights
        hf_state_dict = {}
        model_state_dict = model.state_dict()

        for hf_key, custom_key in state_dict_mapping.items():
            if custom_key in model_state_dict:
                hf_state_dict[hf_key] = model_state_dict[custom_key].clone()
            else:
                print(f"Warning: {custom_key} not found in model state dictionary")

        # Load weights into HF model
        hf_model.load_state_dict(hf_state_dict, strict=False)

        # keep
        os.makedirs(output_dir, exist_ok=True)
        hf_model.save_pretrained(output_dir)
        hf_config.save_pretrained(output_dir)

        print(f"✓ Saved in Hugging Face format: {output_dir}")
        return hf_model, hf_config

    except Exception as e:
        print(f"✗ Error converting to Hugging Face format: {e}")
        return None, None


def load_domain_tokenizer(domain, vocab_path=None):
    """Load domain-specific tokenizer"""
    try:
        if domain == "compounds":
            from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer

            vocab_file = vocab_path or "assets/molecules/vocab.txt"
            if not os.path.exists(vocab_file):
                print(f"Vocabulary file not found: {vocab_file}")
                return None
            return CompoundsTokenizer(vocab_file, 256)

        elif domain == "molecule_nat_lang":
            from molcrawl.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

            return MoleculeNatLangTokenizer()

        elif domain == "genome_sequence":
            from molcrawl.genome_sequence.utils.tokenizer import create_genome_tokenizer

            model_path = vocab_path
            return create_genome_tokenizer(model_path)

        elif domain == "protein_sequence":
            from molcrawl.protein_sequence.utils.bert_tokenizer import (
                create_bert_protein_tokenizer,
            )

            return create_bert_protein_tokenizer()

        elif domain == "rna":
            from molcrawl.rna.utils.bert_tokenizer import create_bert_rna_tokenizer

            return create_bert_rna_tokenizer()

        else:
            print(f"Unknown domain: {domain}")
            return None

    except ImportError as e:
        print(f"Failed to import domain-specific tokenizer: {e}")
        return None
    except Exception as e:
        print(f"Failed to initialize tokenizer: {e}")
        return None


def create_simple_tokenizer(vocab_size):
    """
    Create a simple tokenizer (for debugging)
    """
    try:
        # Create a simple vocabulary
        vocab = {f"<token_{i}>": i for i in range(vocab_size)}
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

        for i, token in enumerate(special_tokens):
            vocab[token] = i

        # Create a simple tokenizer with PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers

        tokenizer_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()

        # HF wrapper
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
        )

        return hf_tokenizer

    except Exception as e:
        print(f"Error creating simple tokenizer: {e}")
        return None


def evaluate_perplexity(model, dataset, tokenizer=None, max_samples=1000, device="cuda"):
    """
    Calculate perplexity on a dataset
    """
    print("\n=== Perplexity evaluation ===")
    print(f"Number of evaluation samples: {min(len(dataset), max_samples)}")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for i in tqdm(range(min(len(dataset), max_samples)), desc="Calculating perplexity"):
            try:
                # get data
                tokens = dataset[i]
                if torch.is_tensor(tokens):
                    input_ids = tokens.unsqueeze(0).to(device)
                else:
                    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

                # Separate input and goal (correct sizing)
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]

                if inputs.size(1) == 0 or targets.size(1) == 0:  # Skip if empty
                    continue

                if inputs.size(1) != targets.size(1):  # Adjust if sizes do not match
                    min_len = min(inputs.size(1), targets.size(1))
                    inputs = inputs[:, :min_len]
                    targets = targets[:, :min_len]

                # Predict with model
                # Pass targets so the custom GPT computes logits for ALL positions.
                # Without targets, the model only forwards the last token (inference
                # optimisation), causing a shape mismatch when computing cross-entropy.
                outputs = model(inputs, targets)
                if isinstance(outputs, tuple):
                    logits, model_loss = outputs
                    if model_loss is not None:
                        # Custom GPT already computed mean cross-entropy loss
                        loss = model_loss
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            targets.reshape(-1),
                            ignore_index=-100,
                        )
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-100,
                    )

                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
                num_batches += 1

            except Exception as e:
                print(f"Error in sample {i}: {e}")
                continue

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        print(f"✓ average loss: {avg_loss:.4f}")
        print(f"✓ perplexity: {perplexity:.4f}")
        print(f"✓ Number of evaluation tokens: {total_tokens:,}")

        return perplexity, avg_loss
    else:
        print("✗ No valid data found")
        return float("inf"), float("inf")


def generate_text_samples(model, tokenizer, device="cuda", num_samples=5, max_length=100):
    """
    Create a text generation sample
    """
    print("\n=== Text generation test ===")

    if tokenizer is None:
        print("Skip text generation because tokenizer is not available")
        return []

    model.eval()
    generated_samples = []

    try:
        for i in range(num_samples):
            # start token
            if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
                start_token = tokenizer.bos_token_id
            else:
                start_token = 0

            input_ids = torch.tensor([[start_token]]).to(device)

            with torch.no_grad():
                # generate
                for _ in range(max_length):
                    outputs = model(input_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                    # predict next token
                    next_token_logits = logits[0, -1, :]
                    next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)

                    # add to input
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

                    # Termination condition
                    if hasattr(tokenizer, "eos_token_id") and next_token.item() == tokenizer.eos_token_id:
                        break

            # decode
            if hasattr(tokenizer, "decode"):
                try:
                    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                except TypeError:
                    # Some domain tokenizers do not support skip_special_tokens
                    generated_text = tokenizer.decode(input_ids[0])
            else:
                generated_text = " ".join([f"token_{tid}" for tid in input_ids[0].tolist()])

            generated_samples.append(generated_text)
            print(f"Sample {i + 1}: {generated_text}")

    except Exception as e:
        print(f"Text generation error: {e}")

    return generated_samples


def calculate_accuracy_metrics(model, dataset, tokenizer=None, max_samples=500, device="cuda"):
    """
    Calculate various accuracy metrics
    """
    print("\n=== Accuracy metrics calculation ===")

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    top5_correct = 0

    with torch.no_grad():
        for i in tqdm(range(min(len(dataset), max_samples)), desc="Calculating accuracy"):
            try:
                tokens = dataset[i]
                if torch.is_tensor(tokens):
                    input_ids = tokens.unsqueeze(0).to(device)
                else:
                    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]

                if inputs.size(1) == 0 or targets.size(1) == 0:  # Skip if empty
                    continue

                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                # Top-1 accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.numel()

                # Top-5 accuracy
                top5_preds = torch.topk(logits, 5, dim=-1).indices
                top5_correct += sum([targets[i] in top5_preds[i] for i in range(len(targets))])

            except Exception:
                continue

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        top5_accuracy = top5_correct / total_predictions

        print(f"✓ Top-1 accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        print(f"✓ Top-5 accuracy: {top5_accuracy:.4f}")

        return accuracy, top5_accuracy
    else:
        print("✗ No valid data found for accuracy calculation")
        return 0.0, 0.0


def test_model_performance(model, model_args, device="cuda"):
    """
    Get model performance statistics
    """
    print("\n=== Model performance statistics ===")

    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of parameters: {total_params:,}")
    print(f"Number of trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"Vocabulary size: {model_args.get('vocab_size', 'Unknown'):,}")
    print(f"Block size: {model_args.get('block_size', 'Unknown')}")
    print(f"Number of layers: {model_args.get('n_layer', 'Unknown')}")
    print(f"Number of heads: {model_args.get('n_head', 'Unknown')}")
    print(f"Embedded dimensions: {model_args.get('n_embd', 'Unknown')}")
    print(f"Device used: {device}")

    # Check GPU usage
    if torch.cuda.is_available() and "cuda" in device:
        print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU memory reservation: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "vocab_size": model_args.get("vocab_size"),
        "block_size": model_args.get("block_size"),
        "n_layer": model_args.get("n_layer"),
        "n_head": model_args.get("n_head"),
        "n_embd": model_args.get("n_embd"),
    }


def generate_test_report(checkpoint_path, results, output_dir):
    """Generate report of test results"""
    report_path = Path(output_dir) / "gpt2_test_report.json"

    # Convert Tensor etc. to serializable format
    def make_serializable(obj):
        if torch.is_tensor(obj):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(item) for item in obj)
        else:
            return obj

    report = {
        "checkpoint_path": checkpoint_path,
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": make_serializable(results),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Test report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive testing and validation script for GPT2 checkpoints")
    parser.add_argument("--checkpoint_path", required=True, help="Path of checkpoint to test")
    parser.add_argument("--output_dir", default="gpt2_test_output", help="output directory")
    parser.add_argument("--convert_to_hf", action="store_true", help="Convert to Hugging Face format")
    parser.add_argument(
        "--test_dataset_params",
        type=str,
        help="Test dataset parameters (JSON format)",
    )
    parser.add_argument(
        "--domain",
        choices=["compounds", "molecule_nat_lang", "genome_sequence", "protein_sequence", "rna"],
        help="Domain to use",
    )
    parser.add_argument("--vocab_path", help="Vocabulary file path")
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to use for testing",
    )
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    print("=== GPT2 checkpoint test/verification script ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")

    results = {}

    # load checkpoint
    model, model_args, iter_num, best_val_loss, config = load_gpt2_checkpoint(args.checkpoint_path, args.device)

    if model is None:
        print("Test is terminating because reading the checkpoint failed.")
        return

    results["checkpoint_info"] = {
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    # Convert to Hugging Face format (optional)
    hf_model = None
    if args.convert_to_hf:
        hf_output_dir = Path(args.output_dir) / "hf_model"
        hf_model, hf_config = convert_to_hf_format(model, model_args, hf_output_dir)
        if hf_model:
            results["hf_conversion"] = "success"
            model = hf_model  # Use converted model for testing
        else:
            results["hf_conversion"] = "failed"

    # load tokenizer (optional)
    tokenizer = None
    if args.domain:
        print(f"\nLoading domain-specific tokenizer: {args.domain}")
        tokenizer = load_domain_tokenizer(args.domain, args.vocab_path)

    if tokenizer is None:
        print("Creating a simple tokenizer...")
        tokenizer = create_simple_tokenizer(model_args["vocab_size"])

    # load test dataset
    test_dataset = None
    if args.test_dataset_params:
        try:
            dataset_params = json.loads(args.test_dataset_params)
            test_dataset = PreparedDataset(**dataset_params, split="valid")
            print(f"✓ Load test dataset: {len(test_dataset)} sample")
        except Exception as e:
            print(f"Error loading test dataset: {e}")

    if test_dataset is None:
        print("Creating default test dataset...")
        # Create a dummy dataset (appropriate size)
        dummy_data = [torch.randint(0, model_args["vocab_size"], (model_args["block_size"],)) for _ in range(100)]
        test_dataset = dummy_data

    try:
        # model performance statistics
        perf_stats = test_model_performance(model, model_args, args.device)
        results["performance_stats"] = perf_stats

        # Perplexity evaluation
        perplexity, avg_loss = evaluate_perplexity(model, test_dataset, tokenizer, args.max_test_samples, args.device)
        results["perplexity"] = perplexity
        results["avg_loss"] = avg_loss

        # Accuracy metrics
        accuracy, top5_accuracy = calculate_accuracy_metrics(model, test_dataset, tokenizer, args.max_test_samples, args.device)
        results["accuracy"] = accuracy
        results["top5_accuracy"] = top5_accuracy

        # Text generation sample
        generated_samples = generate_text_samples(model, tokenizer, args.device)
        results["generated_samples"] = generated_samples

        results["status"] = "success"

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    # generate report
    os.makedirs(args.output_dir, exist_ok=True)
    generate_test_report(args.checkpoint_path, results, args.output_dir)

    print("\n=== Test completed ===")
    print(f"Please check {args.output_dir}/gpt2_test_report.json for detailed results.")


if __name__ == "__main__":
    main()
