"""
Random coefficients + learned routing ablation (Professor Ge Experiment 2).

Uses the standard TaskMap learned routes but replaces mapper-generated
coefficients with fixed random coefficients. Tests whether the coefficients
or the routing carry the adaptation information.

If performance collapses, coefficients are the mechanism.
If performance holds, routing alone is sufficient.

Usage:
  python train_random_coefficients.py --backbone Qwen/Qwen2.5-1.5B --max_steps 6000
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
from losses import TaskMapLossComputer
from train import tokenize_batch, set_seed
from eval import evaluate_task


def parse_args():
    parser = argparse.ArgumentParser(description="Random coefficients ablation")
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
    parser.add_argument("--output_dir", type=str, default="outputs/random_coefficients")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_random_coefficients(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading frozen backbone: {args.backbone}")
    backbone, tokenizer = load_backbone(args.backbone)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    print("\nLoading training data...")
    datasets = {}
    for tid, meta in KNOWN_TASKS.items():
        ds = download_task(tid, meta)
        if ds is not None:
            datasets[tid] = ds
    train_data = format_all_tasks(datasets, split="train")
    task_ids = list(train_data.keys())
    print(f"Tasks: {task_ids}")

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

    print("Computing description embeddings...")
    for tid in task_ids:
        desc = KNOWN_TASKS[tid]["descriptions"][0]
        embed = taskmap.task_code.compute_description_embedding(
            backbone, tokenizer, desc, device
        )
        taskmap.cache_description(tid, embed)

    hook_manager = TaskMapHookManager(backbone, taskmap, block_size=args.block_size)

    # Generate FIXED RANDOM coefficients per task per layer
    print("\nGenerating fixed random coefficients...")
    gen = torch.Generator().manual_seed(args.seed + 2000)
    random_coefficients = {}
    for tid in task_ids:
        task_coeffs = {}
        for l in range(tm_config.num_layers):
            task_coeffs[l] = {
                'c_u': torch.randn(tm_config.num_blocks, tm_config.rank, generator=gen, device=device) * 0.01,
                'c_g': torch.randn(tm_config.num_blocks, tm_config.rank, generator=gen, device=device) * 0.01,
                'c_d': torch.randn(tm_config.num_blocks, tm_config.rank, generator=gen, device=device) * 0.01,
            }
        random_coefficients[tid] = task_coeffs
    print(f"  Generated random coefficients for {len(task_ids)} tasks × {tm_config.num_layers} layers")

    def activate_with_random_coefficients(task_id, device):
        """Use LEARNED routes but RANDOM coefficients."""
        taskmap.clear_route_cache()
        learned_routes = taskmap.compute_route(task_id, device)

        for layer_idx, hook in enumerate(hook_manager.hooks):
            if layer_idx >= tm_config.num_layers:
                break
            route = learned_routes[layer_idx]
            rc = random_coefficients[task_id][layer_idx]
            hook.set_route({
                'mask': route['mask'],
                'selected': route['selected'],
                'c_u': rc['c_u'],
                'c_g': rc['c_g'],
                'c_d': rc['c_d'],
            }, taskmap.residual_bases)

    # Optimizer — train task codes and mapper (routes still learned)
    param_groups = []
    code_params = [p for n, p in taskmap.task_code.named_parameters()
                   if p.requires_grad and "projector" not in n]
    proj_params = [p for n, p in taskmap.task_code.named_parameters()
                   if p.requires_grad and "projector" in n]
    param_groups.append({"params": code_params, "lr": 2e-3})
    param_groups.append({"params": proj_params, "lr": 2e-4})
    mapper_params = [p for p in taskmap.mapper_bank.parameters() if p.requires_grad]
    param_groups.append({"params": mapper_params, "lr": 2e-4})

    optimizer = AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.95))
    max_steps = args.max_steps
    warmup_steps = int(max_steps * 0.03)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(max_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    task_families = {tid: KNOWN_TASKS[tid]["family"] for tid in task_ids}
    loss_computer = TaskMapLossComputer(
        tm_config, FAMILY_PAIRS, task_families,
        lambda_bud=0.05, lambda_topo=0.0, lambda_bal=0.01,
        lambda_stab=1e-3, lambda_sm=1e-3, lambda_align=1e-4,
        active_mapping_loss=False,
    )

    grad_accum = args.gradient_accumulation_steps
    print(f"\nStarting random-coefficients training for {max_steps} steps...")
    print(f"  Routes: LEARNED, Coefficients: FIXED RANDOM")

    global_step = 0
    accum_loss = 0.0
    all_route_masks = {}
    t_start = time.time()

    dataloader = build_dataloader(train_data, args.microbatch_size,
                                  max_steps * grad_accum, args.seed)

    for step_idx, (task_id, examples) in enumerate(dataloader):
        activate_with_random_coefficients(task_id, device)

        taskmap.clear_route_cache()
        routes = taskmap.compute_route(task_id, device)
        masks = [r['mask'].detach() for r in routes]
        all_route_masks[task_id] = masks

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

    print(f"\nTraining complete in {time.time() - t_start:.1f}s")

    # Evaluate
    print("\n=== Random-Coefficients Evaluation ===")
    eval_data = format_all_tasks(datasets, split="validation")
    for tid in eval_data:
        if len(eval_data[tid]) > args.max_eval_examples:
            eval_data[tid] = eval_data[tid][:args.max_eval_examples]

    backbone.eval()
    all_scores = {}
    for tid, examples in eval_data.items():
        if tid not in KNOWN_TASKS:
            continue
        activate_with_random_coefficients(tid, device)
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
    print(f"\n  Random-coefficients macro (excl MBPP): {macro:.2f}")

    results = {
        "mode": "random_coefficients_learned_routing",
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
    train_random_coefficients(args)
