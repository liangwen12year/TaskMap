"""
Random-route baseline: same as TaskMap but with fixed random block selection.

Tests whether learned routing produces better task scores than random routing
at the same active fraction. Uses the same mapper, coefficients, and residual
bases as TaskMap, but routes are frozen random masks.

Usage:
  python train_random_route.py --backbone Qwen/Qwen2.5-1.5B --max_steps 6000
"""

import os
import sys
import time
import json
import argparse
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone
from models.taskmap_model import TaskMapModel, TaskMapConfig
from models.ffn_hooks import TaskMapHookManager
from data.config import KNOWN_TASKS, FAMILY_PAIRS
from data.download import download_task
from data.format import format_all_tasks
from data.sampler import build_dataloader
from train import tokenize_batch, set_seed
from eval import evaluate_task


def parse_args():
    parser = argparse.ArgumentParser(description="Random-route baseline")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--active_fraction", type=float, default=0.75)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--code_dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_eval_examples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="outputs/random_route")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_random_route(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load frozen backbone
    print(f"Loading frozen backbone: {args.backbone}")
    backbone, tokenizer = load_backbone(args.backbone)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Load data
    print("\nLoading training data...")
    datasets = {}
    for tid, meta in KNOWN_TASKS.items():
        ds = download_task(tid, meta)
        if ds is not None:
            datasets[tid] = ds
    train_data = format_all_tasks(datasets, split="train")
    task_ids = list(train_data.keys())
    print(f"Tasks: {task_ids}")

    # Setup TaskMap with unfrozen mapper (same as standard TaskMap)
    tm_config = TaskMapConfig.from_backbone(
        args.backbone,
        block_size=args.block_size,
        active_fraction=args.active_fraction,
        code_dim=args.code_dim,
        rank=args.rank,
        mapper_hidden=512,
        total_steps=args.max_steps,
    )

    taskmap = TaskMapModel(tm_config, num_tasks=len(task_ids), freeze_mapper=False).to(device)
    taskmap.register_tasks(task_ids)

    # Compute description embeddings
    print("Computing description embeddings...")
    for tid in task_ids:
        desc = KNOWN_TASKS[tid]["descriptions"][0]
        embed = taskmap.task_code.compute_description_embedding(
            backbone, tokenizer, desc, device
        )
        taskmap.cache_description(tid, embed)

    # Install hooks
    hook_manager = TaskMapHookManager(backbone, taskmap, block_size=args.block_size)

    # Generate FIXED RANDOM routes per task per layer
    print("\nGenerating fixed random routes...")
    k = max(1, int(args.active_fraction * tm_config.num_blocks))
    gen = torch.Generator().manual_seed(args.seed + 1000)
    random_routes = {}
    for tid in task_ids:
        routes = []
        for l in range(tm_config.num_layers):
            perm = torch.randperm(tm_config.num_blocks, generator=gen)[:k]
            mask = torch.zeros(tm_config.num_blocks, device=device)
            mask[perm] = 1.0
            routes.append({
                'mask': mask,
                'selected': perm.tolist(),
            })
        random_routes[tid] = routes
        print(f"  {tid}: {k} random blocks per layer")

    # Override activate_for_task to use random routes with mapper coefficients
    def activate_random_route(task_id, device):
        """Use random masks but mapper-generated coefficients."""
        # Get coefficients from the mapper (these are learned)
        taskmap.clear_route_cache()
        learned_routes = taskmap.compute_route(task_id, device)

        for layer_idx, hook in enumerate(hook_manager.hooks):
            if layer_idx >= tm_config.num_layers:
                break
            # Use RANDOM mask but LEARNED coefficients
            random_route = random_routes[task_id][layer_idx]
            learned_route = learned_routes[layer_idx]
            hook.set_route({
                'mask': random_route['mask'],
                'selected': random_route['selected'],
                'c_u': learned_route.get('c_u', torch.zeros(tm_config.num_blocks, tm_config.rank, device=device)),
                'c_g': learned_route.get('c_g', torch.zeros(tm_config.num_blocks, tm_config.rank, device=device)),
                'c_d': learned_route.get('c_d', torch.zeros(tm_config.num_blocks, tm_config.rank, device=device)),
            }, taskmap.residual_bases)

    # Optimizer — train task codes and mapper (same as TaskMap)
    param_groups = []
    code_params = []
    projector_params = []
    for name, param in taskmap.task_code.named_parameters():
        if param.requires_grad:
            if "projector" in name:
                projector_params.append(param)
            else:
                code_params.append(param)
    param_groups.append({"params": code_params, "lr": 2e-3})
    param_groups.append({"params": projector_params, "lr": 2e-4})
    mapper_params = [p for p in taskmap.mapper_bank.parameters() if p.requires_grad]
    param_groups.append({"params": mapper_params, "lr": 2e-4})

    optimizer = AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.95))
    max_steps = args.max_steps
    warmup_steps = int(max_steps * 0.03)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(max_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    # Training loop
    grad_accum = args.gradient_accumulation_steps
    print(f"\nStarting random-route training for {max_steps} steps...")
    print(f"  Active fraction: {args.active_fraction} ({k}/{tm_config.num_blocks} blocks, RANDOM)")

    global_step = 0
    accum_loss = 0.0
    t_start = time.time()

    dataloader = build_dataloader(train_data, args.microbatch_size,
                                  max_steps * grad_accum, args.seed)

    for step_idx, (task_id, examples) in enumerate(dataloader):
        activate_random_route(task_id, device)

        batch = tokenize_batch(tokenizer, examples, args.max_seq_length)
        batch = {k_: v.to(device) for k_, v in batch.items()}

        outputs = backbone(**batch)
        loss = outputs.loss / grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (step_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(taskmap.trainable_parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            taskmap.step()
            global_step += 1

            if global_step % 100 == 0 or global_step == 1:
                elapsed = time.time() - t_start
                print(f"  Step {global_step}/{max_steps} | "
                      f"Loss: {accum_loss:.4f} | Task: {task_id} | Time: {elapsed:.0f}s")
                accum_loss = 0.0

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({global_step} steps)")

    # Evaluate
    print("\n=== Random-Route Evaluation ===")
    eval_data = format_all_tasks(datasets, split="validation")
    for tid in eval_data:
        if len(eval_data[tid]) > args.max_eval_examples:
            eval_data[tid] = eval_data[tid][:args.max_eval_examples]

    backbone.eval()
    all_scores = {}
    for tid, examples in eval_data.items():
        if tid not in KNOWN_TASKS:
            continue
        activate_random_route(tid, device)
        scores = evaluate_task(
            backbone, tokenizer, tid, examples,
            KNOWN_TASKS[tid]["metric"], KNOWN_TASKS[tid]["max_response_tokens"], device
        )
        all_scores[tid] = scores
        print(f"    {tid}: {scores}")

    eval_tasks = [t for t in all_scores if t != 'mbpp' and isinstance(all_scores[t], dict)]
    primary = [list(all_scores[t].values())[0] for t in eval_tasks]
    macro = np.mean(primary) if primary else 0.0
    all_scores["macro_avg_9task"] = float(macro)
    print(f"\n  Random-route macro (excl MBPP): {macro:.2f}")

    results = {
        "mode": "random_route",
        "backbone": args.backbone,
        "active_fraction": args.active_fraction,
        "eval_examples": args.max_eval_examples,
        "scores": all_scores,
    }
    print("\n=== RESULTS JSON ===")
    print(json.dumps(results, indent=2, default=str))
    print("=== END RESULTS ===")

    hook_manager.remove_all()


if __name__ == "__main__":
    args = parse_args()
    train_random_route(args)
