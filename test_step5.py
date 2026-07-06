"""
Tests for Step 5: evaluation pipeline, baselines, and table generation.
Runs on CPU with small models/data.

Usage: python test_step5.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from models.taskmap_model import TaskMapConfig
from models.baselines import (
    DirectBlockLoRA, RandomRouteTaskMap, NoDescriptionTaskMap,
    get_method_summary,
)
from eval import accuracy, f1_em, rouge_l, exact_answer, compute_negative_transfer


def test_direct_block_lora():
    print("=" * 60)
    print("TEST 1: Direct Block-LoRA baseline")
    print("=" * 60)
    config = TaskMapConfig(
        num_layers=2, model_dim=16, ffn_dim=32,
        block_size=8, active_fraction=0.5,
        code_dim=8, rank=2, mapper_hidden=16, embed_dim=16,
    )
    tasks = ["sst2", "gsm8k"]
    dbl = DirectBlockLoRA(config, tasks)

    mask, selected, logits = dbl.get_route("sst2", 0)
    c_u, c_g, c_d = dbl.get_coefficients("sst2", 0)
    k = max(1, int(0.5 * config.num_blocks))

    assert mask.sum() == k, f"Expected {k} active blocks, got {mask.sum()}"
    assert c_u.shape == (config.num_blocks, config.rank)

    # All params are trainable (no mapper)
    trainable = dbl.num_trainable()
    assert trainable > 0
    print(f"  Route: {selected}, mask sum: {mask.sum().item()}")
    print(f"  Trainable params: {trainable}")
    print("  PASSED\n")


def test_random_route():
    print("=" * 60)
    print("TEST 2: Random-route mapped baseline")
    print("=" * 60)
    config = TaskMapConfig(
        num_layers=2, model_dim=16, ffn_dim=32,
        block_size=8, active_fraction=0.5,
        code_dim=8, rank=2, mapper_hidden=16, embed_dim=16,
    )
    tasks = ["sst2", "gsm8k"]
    rrm = RandomRouteTaskMap(config, tasks, seed=42)

    mask1, sel1 = rrm.get_route("sst2", 0)
    mask2, sel2 = rrm.get_route("gsm8k", 0)

    k = max(1, int(0.5 * config.num_blocks))
    assert mask1.sum() == k
    assert mask2.sum() == k

    # Same task same layer should give same random route
    mask1b, _ = rrm.get_route("sst2", 0)
    assert torch.equal(mask1, mask1b), "Fixed random route should be deterministic"

    print(f"  sst2 route: {sel1}")
    print(f"  gsm8k route: {sel2}")
    print(f"  Deterministic: {torch.equal(mask1, mask1b)}")
    print("  PASSED\n")


def test_no_description():
    print("=" * 60)
    print("TEST 3: No-description TaskMap baseline")
    print("=" * 60)
    config = TaskMapConfig(
        num_layers=2, model_dim=16, ffn_dim=32,
        block_size=8, active_fraction=0.5,
        code_dim=8, rank=2, mapper_hidden=16, embed_dim=16,
    )
    tasks = ["sst2", "gsm8k"]
    ndt = NoDescriptionTaskMap(config, tasks)

    code_sst2 = ndt.get_code("sst2", 0)
    code_gsm8k = ndt.get_code("gsm8k", 0)

    assert code_sst2.shape == (config.code_dim,)
    assert not torch.equal(code_sst2, code_gsm8k), "Different tasks should have different embeddings"

    # Trainable
    trainable = sum(p.numel() for p in ndt.parameters() if p.requires_grad)
    assert trainable > 0
    print(f"  sst2 code norm: {code_sst2.norm():.4f}")
    print(f"  gsm8k code norm: {code_gsm8k.norm():.4f}")
    print(f"  Trainable params: {trainable}")
    print("  PASSED\n")


def test_negative_transfer():
    print("=" * 60)
    print("TEST 4: Negative transfer computation")
    print("=" * 60)
    multi_scores = {
        "sst2": {"accuracy": 85.0},
        "gsm8k": {"exact_answer": 40.0},
        "xsum": {"rouge_l": 30.0},
    }
    specific_scores = {
        "sst2": {"accuracy": 90.0},    # specific is better -> I_t = 5
        "gsm8k": {"exact_answer": 38.0},  # specific is worse -> I_t = -2
        "xsum": {"rouge_l": 35.0},      # specific is better -> I_t = 5
    }

    result = compute_negative_transfer(multi_scores, specific_scores)
    assert result["per_task"]["sst2"] == 5.0
    assert result["per_task"]["gsm8k"] == -2.0
    assert abs(result["mean_I_t"] - 2.667) < 0.1

    print(f"  Per-task I_t: {result['per_task']}")
    print(f"  Mean I_t: {result['mean_I_t']:.3f}")
    print(f"  Fraction with positive interference: {result['fraction_positive']:.2f}")
    print("  PASSED\n")


def test_method_summary():
    print("=" * 60)
    print("TEST 5: All methods defined for Table 3")
    print("=" * 60)
    methods = get_method_summary()
    expected = [
        "frozen_base", "full_ffn_finetune", "dense_multitask_lora",
        "task_specific_lora", "task_family_lora", "adapter_tuning",
        "direct_block_lora", "random_route_mapped", "no_description_taskmap",
        "taskmap_25", "taskmap_50", "taskmap_75",
    ]
    for m in expected:
        assert m in methods, f"Missing method: {m}"
        print(f"  {m}: {methods[m]}")

    print(f"\n  Total methods: {len(methods)}")
    print("  PASSED\n")


def test_table3_generation():
    print("=" * 60)
    print("TEST 6: Table 3 generation (with placeholders)")
    print("=" * 60)
    # Just verify it runs without error
    from generate_table3 import generate_table
    generate_table("nonexistent_dir")
    print("  PASSED\n")


if __name__ == "__main__":
    test_direct_block_lora()
    test_random_route()
    test_no_description()
    test_negative_transfer()
    test_method_summary()
    test_table3_generation()

    print("=" * 60)
    print("ALL STEP 5 TESTS PASSED")
    print("=" * 60)
    print("\nTo run evaluation:")
    print("  # Dry run (2 tasks, frozen base):")
    print("  python run_eval.py --mode frozen --backbone Qwen/Qwen2.5-0.5B --dry_run")
    print("")
    print("  # Generate Table 3 (after running experiments):")
    print("  python generate_table3.py --results_dir outputs/")
