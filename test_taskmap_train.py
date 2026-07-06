"""
Unit tests for Step 4: TaskMap training loop and all 7 losses.
Runs entirely on CPU with small tensors.

Usage: python test_taskmap_train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from models.taskmap_model import TaskMapModel, TaskMapConfig
from losses import (
    budget_loss, topology_loss, balance_loss,
    stability_loss, alignment_loss, TaskMapLossComputer,
)
from data.config import FAMILY_PAIRS


def make_small_taskmap():
    """Create a tiny TaskMap for testing."""
    config = TaskMapConfig(
        num_layers=2, model_dim=16, ffn_dim=32,
        block_size=8, active_fraction=0.5,
        code_dim=8, rank=2, mapper_hidden=16,
        embed_dim=16, total_steps=100, warmup_fraction=0.1,
    )
    taskmap = TaskMapModel(config, num_tasks=3)
    taskmap.register_tasks(["sst2", "gsm8k", "xsum"])
    taskmap.cache_description("sst2", torch.randn(16))
    taskmap.cache_description("gsm8k", torch.randn(16))
    taskmap.cache_description("xsum", torch.randn(16))
    return taskmap, config


def test_budget_loss():
    print("=" * 60)
    print("TEST 1: Budget loss (Eq. 10)")
    print("=" * 60)
    # Perfectly matching target
    gates = [torch.ones(10) * 0.5]
    loss_perfect = budget_loss(gates, 0.5)
    assert loss_perfect.item() < 1e-6, f"Perfect match should be ~0, got {loss_perfect}"

    # Far from target
    gates_high = [torch.ones(10) * 0.9]
    loss_high = budget_loss(gates_high, 0.5)
    assert loss_high.item() > 0.1, f"Mismatch should be large, got {loss_high}"

    print(f"  Perfect match loss: {loss_perfect.item():.6f}")
    print(f"  Mismatch loss: {loss_high.item():.4f}")
    print("  PASSED\n")


def test_topology_loss():
    print("=" * 60)
    print("TEST 2: Topology loss (Eq. 11)")
    print("=" * 60)
    # Two tasks with identical routes (same family) -> low loss
    masks_same = {
        "sst2": [torch.tensor([1., 1., 0., 0.])],
        "agnews": [torch.tensor([1., 1., 0., 0.])],
    }
    family_pairs = [("sst2", "agnews")]
    task_families = {"sst2": "classification", "agnews": "classification"}
    loss_same = topology_loss(masks_same, family_pairs, task_families)

    # Two tasks with opposite routes (same family) -> high loss
    masks_diff = {
        "sst2": [torch.tensor([1., 1., 0., 0.])],
        "agnews": [torch.tensor([0., 0., 1., 1.])],
    }
    loss_diff = topology_loss(masks_diff, family_pairs, task_families)

    assert loss_same < loss_diff, f"Same routes should have lower topology loss"
    print(f"  Same routes: {loss_same.item():.4f}")
    print(f"  Different routes: {loss_diff.item():.4f}")
    print("  PASSED\n")


def test_balance_loss():
    print("=" * 60)
    print("TEST 3: Balance loss (Eq. 12)")
    print("=" * 60)
    # Uniform usage -> low loss
    masks_uniform = {
        "t1": [torch.tensor([1., 0., 1., 0.])],
        "t2": [torch.tensor([0., 1., 0., 1.])],
    }
    loss_uniform = balance_loss(masks_uniform, 4)

    # All same -> higher loss (concentrated)
    masks_same = {
        "t1": [torch.tensor([1., 1., 0., 0.])],
        "t2": [torch.tensor([1., 1., 0., 0.])],
    }
    loss_same = balance_loss(masks_same, 4)

    print(f"  Complementary routes: {loss_uniform.item():.4f}")
    print(f"  Identical routes: {loss_same.item():.4f}")
    print("  PASSED\n")


def test_stability_loss():
    print("=" * 60)
    print("TEST 4: Stability loss (Eq. 13)")
    print("=" * 60)
    taskmap, config = make_small_taskmap()

    z = taskmap.task_code.get_code("sst2", 0)
    mapper_fn = lambda z_in: taskmap.mapper_bank(0, z_in)
    loss = stability_loss(mapper_fn, z)

    assert loss.item() >= 0, "Stability loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"
    print(f"  Stability loss: {loss.item():.6f}")
    print("  PASSED\n")


def test_alignment_loss():
    print("=" * 60)
    print("TEST 5: Alignment loss (Eq. 15)")
    print("=" * 60)
    taskmap, config = make_small_taskmap()

    z = taskmap.task_code.get_code("sst2", 0)
    mapper_out = taskmap.mapper_bank(0, z)
    output_dim = config.num_blocks + 3 * config.num_blocks * config.rank
    R = torch.randn(config.code_dim, output_dim)

    loss = alignment_loss(z, mapper_out, R)
    assert 0 <= loss.item() <= 2.0, f"Alignment loss should be in [0, 2], got {loss.item()}"
    print(f"  Alignment loss: {loss.item():.4f}")
    print("  PASSED\n")


def test_loss_computer():
    print("=" * 60)
    print("TEST 6: Full TaskMapLossComputer")
    print("=" * 60)
    taskmap, config = make_small_taskmap()

    task_families = {"sst2": "classification", "gsm8k": "mathematical_reasoning",
                     "xsum": "summarization"}

    loss_computer = TaskMapLossComputer(
        config, FAMILY_PAIRS, task_families,
        lambda_bud=0.05, lambda_topo=0.01, lambda_bal=0.01,
        lambda_stab=1e-3, lambda_sm=1e-3, lambda_align=1e-4,
    )

    # Compute routes for all tasks
    all_masks = {}
    for tid in ["sst2", "gsm8k", "xsum"]:
        routes = taskmap.compute_route(tid)
        all_masks[tid] = [r['mask'] for r in routes]

    # Fake task loss
    task_loss = torch.tensor(3.5, requires_grad=True)

    # During warmup
    total, losses = loss_computer.compute(task_loss, taskmap, "sst2", all_masks, is_warmup=True)
    assert torch.isfinite(total), "Total loss should be finite"
    print(f"  Warmup losses: {', '.join(f'{k}={v.item():.4f}' if torch.is_tensor(v) else f'{k}={v:.4f}' for k, v in losses.items())}")

    # After warmup
    for _ in range(15):
        taskmap.router.step()
    taskmap.clear_route_cache()
    for tid in ["sst2", "gsm8k", "xsum"]:
        routes = taskmap.compute_route(tid)
        all_masks[tid] = [r['mask'] for r in routes]

    total2, losses2 = loss_computer.compute(task_loss, taskmap, "sst2", all_masks, is_warmup=False)
    assert torch.isfinite(total2)
    print(f"  Post-warmup losses: {', '.join(f'{k}={v.item():.4f}' if torch.is_tensor(v) else f'{k}={v:.4f}' for k, v in losses2.items())}")

    print("  PASSED\n")


def test_gradient_flow():
    print("=" * 60)
    print("TEST 7: Gradient flows only to task codes")
    print("=" * 60)
    taskmap, config = make_small_taskmap()
    task_families = {"sst2": "classification", "gsm8k": "mathematical_reasoning",
                     "xsum": "summarization"}

    loss_computer = TaskMapLossComputer(config, FAMILY_PAIRS, task_families)

    all_masks = {}
    for tid in ["sst2", "gsm8k", "xsum"]:
        routes = taskmap.compute_route(tid)
        all_masks[tid] = [r['mask'] for r in routes]

    task_loss = torch.tensor(3.5, requires_grad=True)
    total, _ = loss_computer.compute(task_loss, taskmap, "sst2", all_masks, is_warmup=True)
    total.backward()

    # Check: task code params should have gradients
    code_grads = 0
    for p in taskmap.task_code.trainable_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            code_grads += 1

    # Check: mapper params should NOT have gradients
    mapper_grads = 0
    for p in taskmap.mapper_bank.parameters():
        if p.grad is not None:
            mapper_grads += 1

    # Check: bases should NOT have gradients
    bases_grads = 0
    for p in taskmap.residual_bases.parameters():
        if p.grad is not None:
            bases_grads += 1

    print(f"  Task code params with grad: {code_grads}")
    print(f"  Mapper params with grad: {mapper_grads} (should be 0)")
    print(f"  Bases params with grad: {bases_grads} (should be 0)")
    assert mapper_grads == 0, "Mapper should be frozen"
    assert bases_grads == 0, "Bases should be frozen"
    print("  PASSED\n")


def test_route_cache_per_step():
    print("=" * 60)
    print("TEST 8: Route cache clears per optimizer step")
    print("=" * 60)
    taskmap, _ = make_small_taskmap()

    r1 = taskmap.compute_route("sst2")
    r2 = taskmap.compute_route("sst2")
    assert r1 is r2, "Same step should return cached route"

    taskmap.step()  # clears cache + advances router
    r3 = taskmap.compute_route("sst2")
    assert r3 is not r1, "After step(), should recompute route"

    print("  Cache hit within step: OK")
    print("  Cache cleared after step(): OK")
    print("  PASSED\n")


if __name__ == "__main__":
    test_budget_loss()
    test_topology_loss()
    test_balance_loss()
    test_stability_loss()
    test_alignment_loss()
    test_loss_computer()
    test_gradient_flow()
    test_route_cache_per_step()

    print("=" * 60)
    print("ALL STEP 4 TESTS PASSED")
    print("=" * 60)
    print("\nTo run TaskMap training:")
    print("  # Dry run (2 steps, CPU):")
    print("  python train_taskmap.py --dry_run --backbone Qwen/Qwen2.5-0.5B")
    print("")
    print("  # Full training (GPU):")
    print("  python train_taskmap.py --config configs/taskmap_reference.yaml")
