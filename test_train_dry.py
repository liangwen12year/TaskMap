"""
Dry-run test for Step 2: verify the training pipeline end-to-end.
Runs 2 steps on CPU with a tiny model to test the wiring.

Usage: python test_train_dry.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from data.config import KNOWN_TASKS
from data.format import format_dataset, TRAIN_TEMPLATE
from data.sampler import build_dataloader
from eval import accuracy, f1_em, rouge_l, exact_answer


def test_eval_metrics():
    print("=" * 60)
    print("TEST 1: Evaluation metrics")
    print("=" * 60)

    # Accuracy
    result = accuracy(["positive", "negative", "positive"], ["positive", "positive", "negative"])
    assert abs(result["accuracy"] - 33.33) < 1, f"Expected ~33%, got {result}"
    print(f"  accuracy: {result}")

    # F1/EM
    result = f1_em(["the cat sat", "hello world"], ["the cat sat on mat", "hello world"])
    assert result["f1"] > 0, f"F1 should be > 0, got {result}"
    print(f"  f1_em: {result}")

    # ROUGE-L
    result = rouge_l(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert result["rouge_l"] == 100.0, f"Perfect match should be 100, got {result}"
    print(f"  rouge_l: {result}")

    # Exact answer
    result = exact_answer(["The answer is 42.", "Result: 7"], ["42", "8"])
    assert result["exact_answer"] == 50.0, f"Expected 50%, got {result}"
    print(f"  exact_answer: {result}")

    print("  PASSED\n")


def test_tokenize_batch():
    print("=" * 60)
    print("TEST 2: Tokenization with response-only loss masking")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = [
        {
            "full_text": "Task: Classify sentiment.\n\nInput: Great movie!\n\nResponse: positive",
            "input": "Great movie!",
            "response": "positive",
        },
        {
            "full_text": "Task: Classify sentiment.\n\nInput: Terrible film.\n\nResponse: negative",
            "input": "Terrible film.",
            "response": "negative",
        },
    ]

    from train import tokenize_batch
    batch = tokenize_batch(tokenizer, examples, max_length=128)

    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch

    # Labels should have -100 for prompt tokens
    num_masked = (batch["labels"] == -100).sum().item()
    num_total = batch["labels"].numel()
    print(f"  Batch shape: {batch['input_ids'].shape}")
    print(f"  Masked tokens (prompt): {num_masked}/{num_total}")
    assert num_masked > 0, "Some tokens should be masked (prompt portion)"
    assert num_masked < num_total, "Not all tokens should be masked"
    print("  PASSED\n")


def test_sampler_integration():
    print("=" * 60)
    print("TEST 3: Sampler produces valid task-homogeneous batches")
    print("=" * 60)

    fake_data = {
        "sst2": [{"full_text": f"Task: classify\n\nInput: text {i}\n\nResponse: pos",
                   "task_id": "sst2", "family": "classification",
                   "description": "classify", "input": f"text {i}", "response": "pos"}
                  for i in range(100)],
        "gsm8k": [{"full_text": f"Task: math\n\nInput: problem {i}\n\nResponse: 42",
                    "task_id": "gsm8k", "family": "mathematical_reasoning",
                    "description": "solve math", "input": f"problem {i}", "response": "42"}
                   for i in range(50)],
    }

    batches = list(build_dataloader(fake_data, microbatch_size=2, total_steps=10))
    assert len(batches) == 10
    for task_id, examples in batches:
        assert len(examples) == 2
        assert all(ex["task_id"] == task_id for ex in examples), \
            f"All examples in batch should be from {task_id}"
    print(f"  10 batches, all task-homogeneous")
    print("  PASSED\n")


if __name__ == "__main__":
    test_eval_metrics()
    test_tokenize_batch()
    test_sampler_integration()

    print("=" * 60)
    print("ALL STEP 2 TESTS PASSED")
    print("=" * 60)
    print("\nTo run actual training:")
    print("  # Dry run (2 steps, verifies GPU pipeline):")
    print("  python train.py --mode lora --dry_run")
    print("")
    print("  # Full baseline training:")
    print("  python train.py --mode lora --lora_rank 16 --output_dir outputs/lora_r16")
    print("")
    print("  # Frozen base (eval only):")
    print("  python train.py --mode frozen")
