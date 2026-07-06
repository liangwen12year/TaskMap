"""
Unit tests for Step 3: TaskMap core architecture.
Verifies all components without requiring a GPU.

Usage: python test_taskmap_arch.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from models.task_code import TaskCodeModule
from models.mapper import FrozenMapper, MapperBank
from models.router import TopKRouter, hard_topk, gumbel_soft_topk
from models.block_residuals import BlockResidualBases
from models.taskmap_model import TaskMapModel, TaskMapConfig


def test_task_code():
    print("=" * 60)
    print("TEST 1: Task code module")
    print("=" * 60)
    num_layers, embed_dim, code_dim = 4, 64, 16
    tc = TaskCodeModule(num_layers, embed_dim, code_dim, num_tasks=3)
    tc.register_tasks(["sst2", "gsm8k", "xsum"])

    fake_embed = torch.randn(embed_dim)
    tc.cache_description_embedding("sst2", fake_embed)
    tc.cache_description_embedding("gsm8k", torch.randn(embed_dim))

    z = tc.get_code("sst2", layer_idx=0)
    assert z.shape == (code_dim,), f"Expected ({code_dim},), got {z.shape}"

    all_codes = tc.get_all_layer_codes("sst2")
    assert len(all_codes) == num_layers

    z_cold = tc.get_code("gsm8k", layer_idx=0)
    assert z_cold.shape == (code_dim,)

    print(f"  Code shape: {z.shape}")
    print(f"  Trainable params: {tc.num_trainable()}")
    print(f"  Codes per layer differ: {not torch.allclose(all_codes[0], all_codes[1])}")
    print("  PASSED\n")


def test_mapper():
    print("=" * 60)
    print("TEST 2: Frozen modulated mapper")
    print("=" * 60)
    code_dim, num_blocks, rank = 16, 10, 4
    mapper = FrozenMapper(code_dim, num_blocks, rank, hidden_dim=64)
    mapper.freeze()

    grad_count = sum(1 for p in mapper.parameters() if p.requires_grad)
    assert grad_count == 0, f"Mapper should be frozen, but {grad_count} params have grad"

    z = torch.randn(code_dim)
    q, c_u, c_g, c_d = mapper(z)

    assert q.shape == (num_blocks,), f"Route logits: expected ({num_blocks},), got {q.shape}"
    assert c_u.shape == (num_blocks, rank), f"c_u: expected ({num_blocks},{rank}), got {c_u.shape}"
    assert c_g.shape == (num_blocks, rank)
    assert c_d.shape == (num_blocks, rank)

    print(f"  Route logits shape: {q.shape}")
    print(f"  Coefficient shapes: c_u={c_u.shape}, c_g={c_g.shape}, c_d={c_d.shape}")
    print(f"  All params frozen: {grad_count == 0}")
    print("  PASSED\n")


def test_mapper_bank():
    print("=" * 60)
    print("TEST 3: Mapper bank (all layers)")
    print("=" * 60)
    num_layers, code_dim, num_blocks, rank = 4, 16, 10, 4
    bank = MapperBank(num_layers, code_dim, num_blocks, rank, hidden_dim=64)

    codes = [torch.randn(code_dim) for _ in range(num_layers)]
    outputs = bank.forward_all_layers(codes)
    assert len(outputs) == num_layers
    for q, c_u, c_g, c_d in outputs:
        assert q.shape == (num_blocks,)

    print(f"  {num_layers} mappers, all frozen")
    print("  PASSED\n")


def test_router():
    print("=" * 60)
    print("TEST 4: Top-k router with schedule")
    print("=" * 60)
    num_blocks, k = 10, 3
    router = TopKRouter(num_blocks, active_fraction=0.3, warmup_fraction=0.1,
                        total_steps=100)

    # During warmup: all blocks active
    logits = torch.randn(num_blocks)
    mask, selected = router.route(logits)
    assert mask.sum() == num_blocks, f"During warmup, all blocks should be active, got {mask.sum()}"
    print(f"  Warmup (step {router.current_step}): {mask.sum().item():.0f} active (all)")

    # Advance past warmup
    for _ in range(15):
        router.step()
    mask, selected = router.route(logits)
    assert len(selected) == k, f"After warmup, should select {k} blocks, got {len(selected)}"
    print(f"  Post-warmup (step {router.current_step}): {len(selected)} selected, temp={router.get_temperature():.3f}")

    # Advance to end
    for _ in range(85):
        router.step()
    mask, selected = router.route(logits)
    print(f"  Final (step {router.current_step}): {len(selected)} selected, temp={router.get_temperature():.3f}")

    print("  PASSED\n")


def test_hard_topk_gradient():
    print("=" * 60)
    print("TEST 5: Hard top-k has straight-through gradient")
    print("=" * 60)
    logits = torch.randn(10, requires_grad=True)
    mask = hard_topk(logits, k=3)
    loss = (mask * torch.randn(10)).sum()
    loss.backward()

    assert logits.grad is not None, "Gradient should flow through straight-through"
    assert (mask.detach() != 0).sum() == 3, "Exactly 3 blocks should be selected"
    print(f"  Gradient exists: {logits.grad is not None}")
    print(f"  Gradient shape: {logits.grad.shape}")
    print(f"  Non-zero mask entries: {(mask.detach() != 0).sum().item()}")
    print("  PASSED\n")


def test_block_residuals():
    print("=" * 60)
    print("TEST 6: Block-local low-rank residuals")
    print("=" * 60)
    num_layers, num_blocks, rank = 2, 5, 4
    model_dim, block_size = 32, 8
    bases = BlockResidualBases(num_layers, num_blocks, rank, model_dim, block_size)

    grad_count = sum(1 for p in bases.parameters() if p.requires_grad)
    assert grad_count == 0, f"Bases should be frozen, got {grad_count} trainable"

    coeffs = torch.randn(rank)
    delta_u = bases.compute_residual(0, 0, 'u', coeffs)
    assert delta_u.shape == (model_dim, block_size), f"Up residual: expected ({model_dim},{block_size}), got {delta_u.shape}"

    delta_d = bases.compute_residual(0, 0, 'd', coeffs)
    assert delta_d.shape == (block_size, model_dim), f"Down residual: expected ({block_size},{model_dim}), got {delta_d.shape}"

    print(f"  delta_W^u shape: {delta_u.shape}")
    print(f"  delta_W^d shape: {delta_d.shape}")
    print(f"  All bases frozen: {grad_count == 0}")
    print(f"  Memory: {bases.memory_bytes() / 1e6:.2f} MB")
    print("  PASSED\n")


def test_full_taskmap():
    print("=" * 60)
    print("TEST 7: Full TaskMap model assembly")
    print("=" * 60)
    config = TaskMapConfig(
        num_layers=4, model_dim=32, ffn_dim=64,
        block_size=8, active_fraction=0.5,
        code_dim=16, rank=4, mapper_hidden=32,
        embed_dim=32, total_steps=100,
    )
    taskmap = TaskMapModel(config, num_tasks=3)
    taskmap.register_tasks(["sst2", "gsm8k", "xsum"])
    taskmap.cache_description("sst2", torch.randn(32))
    taskmap.cache_description("gsm8k", torch.randn(32))

    routes = taskmap.compute_route("sst2")
    assert len(routes) == config.num_layers
    for l, r in enumerate(routes):
        assert 'mask' in r
        assert 'selected' in r
        assert 'c_u' in r
        print(f"  Layer {l}: mask sum={r['mask'].sum():.1f}, selected={r['selected']}")

    summary = taskmap.parameter_summary()
    print(f"  Parameter summary: {summary}")

    # Verify only task codes are trainable
    trainable_params = taskmap.trainable_parameters()
    assert len(trainable_params) > 0, "Should have trainable parameters"

    mapper_grads = [p.requires_grad for p in taskmap.mapper_bank.parameters()]
    assert not any(mapper_grads), "Mapper should be frozen"

    bases_grads = [p.requires_grad for p in taskmap.residual_bases.parameters()]
    assert not any(bases_grads), "Bases should be frozen"

    print("  PASSED\n")


def test_cold_start():
    print("=" * 60)
    print("TEST 8: Cold-start routing (r=0, description only)")
    print("=" * 60)
    config = TaskMapConfig(
        num_layers=4, model_dim=32, ffn_dim=64,
        block_size=8, active_fraction=0.5,
        code_dim=16, rank=4, mapper_hidden=32,
        embed_dim=32, total_steps=100,
    )
    taskmap = TaskMapModel(config, num_tasks=2)
    taskmap.register_tasks(["sst2", "gsm8k"])

    # Cold-start: task not registered, only description embedding
    taskmap.cache_description("unseen_task", torch.randn(32))

    # Advance past warmup so routing is active
    for _ in range(10):
        taskmap.router.step()

    routes = taskmap.compute_route("unseen_task")
    assert len(routes) == config.num_layers
    for r in routes:
        assert len(r['selected']) > 0, "Cold-start should still produce a route"

    print(f"  Cold-start route for unseen task: {[r['selected'] for r in routes]}")
    print("  PASSED\n")


def test_dense_mask_equivalence():
    print("=" * 60)
    print("TEST 9: Route cache works correctly")
    print("=" * 60)
    config = TaskMapConfig(
        num_layers=2, model_dim=16, ffn_dim=32,
        block_size=8, active_fraction=0.5,
        code_dim=8, rank=2, mapper_hidden=16,
        embed_dim=16, total_steps=100,
    )
    taskmap = TaskMapModel(config, num_tasks=2)
    taskmap.register_tasks(["sst2", "gsm8k"])
    taskmap.cache_description("sst2", torch.randn(16))

    # Compute route twice — should be cached
    routes1 = taskmap.compute_route("sst2")
    routes2 = taskmap.compute_route("sst2")
    assert routes1 is routes2, "Route should be cached"

    # Clear cache and recompute
    taskmap.clear_route_cache()
    routes3 = taskmap.compute_route("sst2")
    assert routes3 is not routes1, "After cache clear, should recompute"

    print("  Cache hit: same object returned")
    print("  Cache clear: new object created")
    print("  PASSED\n")


if __name__ == "__main__":
    test_task_code()
    test_mapper()
    test_mapper_bank()
    test_router()
    test_hard_topk_gradient()
    test_block_residuals()
    test_full_taskmap()
    test_cold_start()
    test_dense_mask_equivalence()

    print("=" * 60)
    print("ALL STEP 3 TESTS PASSED")
    print("=" * 60)
