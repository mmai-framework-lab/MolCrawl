"""
Comprehensive test script for BERT checkpoints

This script runs the following tests:
1. Model and tokenizer loading test
2. Testing the masked language model (MLM)
3. Embedding generation test
4. Batch processing test
5. Model performance statistics
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from datasets import load_from_disk
from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling


def load_domain_tokenizer(domain, vocab_path=None):
    """Load domain-specific tokenizer"""
    try:
        if domain == "compounds":
            from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer

            vocab_file = vocab_path or "assets/molecules/vocab.txt"
            if not os.path.exists(vocab_file):
                print(f"Vocabulary file not found: {vocab_file}")
                return None
            return CompoundsTokenizer(vocab_file, 256)

        elif domain == "molecule_nat_lang":
            # BERT-compatible tokenizer for molecular-related natural languages
            from molcrawl.data.molecule_nat_lang.utils.bert_tokenizer import (
                create_bert_molecule_nat_lang_tokenizer,
            )

            return create_bert_molecule_nat_lang_tokenizer()

        elif domain == "genome_sequence":
            # SentencePiece tokenizer for genome sequences
            from molcrawl.data.genome_sequence.utils.tokenizer import create_genome_tokenizer

            model_path = vocab_path  # SentencePiece model file path
            return create_genome_tokenizer(model_path)

        elif domain == "protein_sequence":
            # BERT compatible ESM tokenizer for protein sequences
            from molcrawl.data.protein_sequence.utils.bert_tokenizer import (
                create_bert_protein_tokenizer,
            )

            return create_bert_protein_tokenizer()

        elif domain == "rna":
            # BERT compatible tokenizer for RNA sequences
            from molcrawl.data.rna.utils.bert_tokenizer import create_bert_rna_tokenizer

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


def load_model_and_tokenizer(checkpoint_path, domain=None, vocab_path=None):
    """Load BERT model and tokenizer from checkpoint"""
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = BertForMaskedLM.from_pretrained(checkpoint_path)

        tokenizer = None

        # First try loading tokenizer from checkpoint
        try:
            tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
            print("✓ Loaded tokenizer from checkpoint")
        except Exception as e:
            print(f"Failed to load tokenizer from checkpoint: {e}")

        if tokenizer is None and domain:
            print(f"Loading domain-specific tokenizer: {domain}")
            tokenizer = load_domain_tokenizer(domain, vocab_path)
            if tokenizer:
                print(f"✓ Loaded tokenizer for {domain}")

        # If no tokenizer can be loaded
        if tokenizer is None:
            print("⚠ Tokenizer not available. Some tests will be skipped.")

        print("✓ Model loaded successfully")
        return model, tokenizer

    except Exception as e:
        print(f"✗ An error occurred while loading the model: {e}")
        return None, None


def safe_convert_tokens_to_string(tokenizer: Any, tokens: Sequence[str]) -> str:
    """Safely restore token string to string (fast tokenizer countermeasure when decoder is not set)"""
    if hasattr(tokenizer, "convert_tokens_to_string"):
        try:
            return tokenizer.convert_tokens_to_string(list(tokens))
        except Exception:
            # Fallback as it will be an exception if the decoder is not set for fast tokenizer
            pass
    # Join without spaces for character-wise tokenizers
    return "".join(tokens)


def test_basic_functionality(model, tokenizer, test_texts):
    """Testing basic functionality"""
    print("\n=== Basic functionality test ===")

    if tokenizer is None:
        print("Skipping basic functionality tests because tokenizer is not available")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for i, text in enumerate(test_texts):
        print(f"\nTest {i + 1}: {text}")

        try:
            # For BERT compatible tokenizers (standard BertTokenizer or custom wrapper)
            if callable(tokenizer) and hasattr(tokenizer, "model_input_names"):
                # BERT compatible wrappers (RNA, Protein, Genome, etc.)
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                print(f" ✓ Inference successful - output shape: {outputs.logits.shape}")

            # For domain-specific tokenizers
            elif hasattr(tokenizer, "tokenize_text"):
                # For CompoundsTokenizer etc.
                if hasattr(tokenizer, "encode"):
                    input_ids = tokenizer.encode(text)
                    input_ids = torch.tensor([input_ids]).to(device)

                    with torch.no_grad():
                        outputs = model(input_ids)

                    print(f" ✓ Inference successful - output shape: {outputs.logits.shape}")
                else:
                    print(" ! Encoding feature not supported")
            else:
                # For standard BertTokenizer
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                print(f" ✓ Inference successful - output shape: {outputs.logits.shape}")

        except Exception as e:
            print(f" ✗ Inference error: {e}")


def test_masked_language_modeling(model, tokenizer, test_texts):
    """Testing masked language modeling"""
    print("\n=== Mask language modeling test ===")

    if tokenizer is None:
        print("Skip MLM test because tokenizer is not available")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for text in test_texts[:2]:  # test with the first two texts
        try:
            # For BERT compatible tokenizers (standard or custom wrappers)
            if callable(tokenizer) and hasattr(tokenizer, "model_input_names"):
                # For standard BertTokenizer or BERT compatible wrapper
                tokens = tokenizer.tokenize(text)
                if len(tokens) > 3:
                    # mask intermediate tokens
                    mask_idx = len(tokens) // 2
                    original_token = tokens[mask_idx]
                    tokens[mask_idx] = tokenizer.mask_token if hasattr(tokenizer, "mask_token") else "[MASK]"

                    # Rebuild string from token (measure against fast tokenizer when decoder is not set)
                    masked_text = safe_convert_tokens_to_string(tokenizer, tokens)

                    print(f"\nOriginal text: {text}")
                    print(f"Masked text: {masked_text}")

                    # predictionexecution
                    inputs = tokenizer(
                        masked_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Get predicted mask position
                    mask_token_id = getattr(tokenizer, "mask_token_id", 4)  # Default is 4
                    mask_token_index = torch.where(inputs["input_ids"] == mask_token_id)[1]
                    if len(mask_token_index) > 0:
                        mask_token_logits = outputs.logits[0, mask_token_index[0], :]
                        top_5_tokens = torch.topk(mask_token_logits, 5, dim=-1)

                        print(f"Original token: {original_token}")
                        print("Predicted top 5 tokens:")
                        for i, (score, token_id) in enumerate(zip(top_5_tokens.values, top_5_tokens.indices)):
                            if hasattr(tokenizer, "decode"):
                                token = tokenizer.decode([token_id], skip_special_tokens=True)
                            else:
                                token = f"Token_{token_id.item()}"
                            print(f" {i + 1}. {token} (score: {score.item():.3f})")

            # For domain-specific tokenizers
            elif hasattr(tokenizer, "tokenize_text"):
                # For CompoundsTokenizer etc.
                tokens = tokenizer.tokenize(text)
                if len(tokens) > 3:
                    # mask intermediate tokens
                    mask_idx = len(tokens) // 2
                    original_token = tokens[mask_idx]
                    tokens[mask_idx] = "[MASK]"  # Mask token

                    # convert token back to string
                    masked_text = "".join(tokens)

                    print(f"\nOriginal text: {text}")
                    print(f"Masked text: {masked_text}")

                    # encode
                    try:
                        if hasattr(tokenizer, "encode"):
                            input_ids = tokenizer.encode(masked_text)
                            input_ids = torch.tensor([input_ids]).to(device)
                        else:
                            print("Encoding feature not supported")
                            continue

                        with torch.no_grad():
                            outputs = model(input_ids)

                        # Find mask position
                        mask_token_id = tokenizer.vocab.get("[MASK]", None)
                        if mask_token_id is not None:
                            mask_positions = torch.where(input_ids == mask_token_id)[1]
                            if len(mask_positions) > 0:
                                mask_token_logits = outputs.logits[0, mask_positions[0], :]
                                top_5_tokens = torch.topk(mask_token_logits, 5, dim=-1)

                                print(f"Original token: {original_token}")
                                print("Predicted top 5 tokens:")
                                for i, (score, token_id) in enumerate(zip(top_5_tokens.values, top_5_tokens.indices)):
                                    token = tokenizer.ids_to_tokens.get(token_id.item(), "[UNK]")
                                    print(f" {i + 1}. {token} (score: {score.item():.3f})")

                    except Exception as e:
                        print(f"Encoding/inference error: {e}")

            else:
                # For standard BertTokenizer
                tokens = tokenizer.tokenize(text)
                if len(tokens) > 3:
                    # mask intermediate tokens
                    mask_idx = len(tokens) // 2
                    original_token = tokens[mask_idx]
                    tokens[mask_idx] = tokenizer.mask_token
                    masked_text = tokenizer.convert_tokens_to_string(tokens)

                    print(f"\nOriginal text: {text}")
                    print(f"Masked text: {masked_text}")

                    # predictionexecution
                    inputs = tokenizer(masked_text, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Get predicted mask position
                    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
                    if len(mask_token_index) > 0:
                        mask_token_logits = outputs.logits[0, mask_token_index[0], :]
                        top_5_tokens = torch.topk(mask_token_logits, 5, dim=-1)

                        print(f"Original token: {original_token}")
                        print("Predicted top 5 tokens:")
                        for i, (score, token_id) in enumerate(zip(top_5_tokens.values, top_5_tokens.indices)):
                            token = tokenizer.decode([token_id])
                            print(f" {i + 1}. {token} (score: {score.item():.3f})")

        except Exception as e:
            print(f"Error during MLM test: {e}")
            continue


def test_embedding_generation(model, tokenizer, test_texts):
    """Testing embedding generation"""
    print("\n=== Embedding generation test ===")

    if tokenizer is None:
        print("Skip embedding test because tokenizer is not available")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    embeddings = []

    for text in test_texts[:3]:  # test with first 3 texts
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.bert(**inputs)  # For BertForMaskedLM, access the bert layer
                # [CLS] Get token embedding
                cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                embeddings.append(cls_embedding)

            print(f"✓ Embedding generation successful - shape: {cls_embedding.shape}")

        except Exception as e:
            print(f"Embedding generation error: {e}")
            continue

    if len(embeddings) > 1:
        print("\nCosine similarity between embeddings:")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f" text{i + 1} vs text{j + 1}: {similarity:.3f}")


def test_batch_processing(model, tokenizer, test_texts):
    """Testing batch processing"""
    print("\n=== Batch processing test ===")

    if tokenizer is None:
        print("Skip batch processing test because tokenizer is not available")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Test by changing batch size
    batch_sizes = [1, 2, 4]

    for batch_size in batch_sizes:
        if batch_size > len(test_texts):
            continue

        batch_texts = test_texts[:batch_size]

        try:
            start_time = time.time()
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs)

            end_time = time.time()
            processing_time = end_time - start_time

            print(f"✓ Batch size {batch_size}: {processing_time:.3f} seconds")

        except Exception as e:
            print(f"✗ Batch size {batch_size}: Error - {e}")


def test_model_performance(model, tokenizer, dataset_path=None):
    """Model performance testing"""
    print("\n=== Performance test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Display model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of parameters: {total_params:,}")
    print(f"Number of trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"Device used: {device}")

    # Check GPU usage（CUDAif available)
    if torch.cuda.is_available():
        print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU memory reservation: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

    # Evaluation on dataset (if available)
    if dataset_path and tokenizer:
        try:
            print(f"\nEvaluation on dataset: {dataset_path}")
            dataset = load_from_disk(dataset_path)

            if "test" in dataset:
                test_dataset = dataset["test"].select(range(min(100, dataset["test"].num_rows)))
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

                model.eval()
                total_loss = 0
                num_batches = 0

                for i in range(0, len(test_dataset), 8):  # batch size 8
                    batch = test_dataset[i : i + 8]
                    batch = data_collator([batch[j] for j in range(len(batch))])
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.no_grad():
                        outputs = model(**batch)
                        total_loss += outputs.loss.item()
                        num_batches += 1

                avg_loss = total_loss / num_batches
                perplexity = torch.exp(torch.tensor(avg_loss))
                print(f"average loss: {avg_loss:.4f}")
                print(f"perplexity: {perplexity:.4f}")

        except Exception as e:
            print(f"Error evaluating dataset: {e}")


def generate_test_report(checkpoint_path, results):
    """Generate report of test results"""
    report_path = Path(checkpoint_path).parent / "test_report.json"

    report = {
        "checkpoint_path": checkpoint_path,
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Test report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BERT checkpoint comprehensive test script")
    parser.add_argument("--checkpoint_path", required=True, help="Path of checkpoint to test")
    parser.add_argument("--dataset_path", help="Evaluation dataset path (optional)")
    parser.add_argument(
        "--domain",
        choices=["compounds", "molecule_nat_lang", "genome_sequence", "protein_sequence", "rna"],
        help="Domain to use (compounds, molecule_nat_lang, genome_sequence, protein_sequence, rna)",
    )
    parser.add_argument(
        "--vocab_path",
        help="Vocabulary file path (compounds: vocab.txt, genome_sequence: spm_tokenizer.model)",
    )
    parser.add_argument(
        "--test_texts",
        nargs="*",
        default=None,
        help="Sample text for testing",
    )

    args = parser.parse_args()

    # set domain-specific default test text
    if args.test_texts is None:
        if args.domain == "compounds":
            args.test_texts = [
                "CC(C)C",
                "CCO",
                "CC(=O)O",
                "C1=CC=CC=C1",
                "CCCCCCCCCCCCCCCCCC(=O)O",
            ]
        elif args.domain == "protein_sequence":
            args.test_texts = [
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK",
                "MHHHHHHSSGVDLGTENLYFQSMKTFRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDI",
                "AKLRDPSFDENIQKALKIAKQLQAEKQAKKQVEQIKLKQQKQVKLAKQEAKLQELQEKLQAKK",
                "MGSSHHHHHHSSGLVPRGSHMKTFRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIL",
            ]
        elif args.domain == "genome_sequence":
            args.test_texts = [
                "ATCGATCGATCGATCG",
                "GCTAGCTAGCTAGCTA",
                "TTTTAAAACCCCGGGG",
                "AAATTTCCCGGGAAATTTCCCGGG",
                "ATGCATGCATGCATGC",
            ]
        elif args.domain == "rna":
            args.test_texts = [
                "cell_type_B_cell",
                "cell_type_T_cell",
                "tissue_brain",
                "tissue_liver",
                "gene_expression_high",
            ]
        elif args.domain == "molecule_nat_lang":
            args.test_texts = [
                "This molecule shows anticancer activity.",
                "Measuring the effectiveness of drugs.",
                "Molecular structure analysis in progress.",
                "Mechanism of chemical reactions.",
                "Identification of biomarkers.",
            ]
        else:
            # default
            args.test_texts = [
                "This is a test sample.",
                "Analyze the structure of molecules.",
                "Evaluating the performance of machine learning models.",
                "Natural language processing technology is progressing.",
                "Data science is an important field.",
            ]

    print("=== BERT checkpoint test script ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    if args.domain:
        print(f"Domain: {args.domain}")
    if args.vocab_path:
        print(f"Vocabulary file: {args.vocab_path}")

    results = {}

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path, args.domain, args.vocab_path)

    if model is None:
        print("Test will end because model loading failed.")
        return

    try:
        # Execute various tests
        test_basic_functionality(model, tokenizer, args.test_texts)
        test_masked_language_modeling(model, tokenizer, args.test_texts)
        test_embedding_generation(model, tokenizer, args.test_texts)
        test_batch_processing(model, tokenizer, args.test_texts)
        test_model_performance(model, tokenizer, args.dataset_path)

        results["status"] = "success"
        results["tests_completed"] = [
            "basic_functionality",
            "mlm",
            "embedding",
            "batch_processing",
            "performance",
        ]

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    # generate report
    generate_test_report(args.checkpoint_path, results)

    print("\n=== Test completed ===")


if __name__ == "__main__":
    main()
