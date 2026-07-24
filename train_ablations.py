"""
Ablation experiments for Professor Ge's suggestions.

Exp 3: Shared coefficients — one coefficient vector per layer, not per block
Exp 6: Linear mapper — replace modulated mapper with single linear layer
Exp 7: Layer selection — adapt only top/middle/bottom layers

Usage:
  python train_ablations.py --ablation shared_coefficients
  python train_ablations.py --ablation linear_mapper
  python train_ablations.py --ablation top_layers_only
  python train_ablations.py --ablation middle_layers_only
  python train_ablations.py --ablation bottom_layers_only
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
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
    parser = argparse.ArgumentParser(description="TaskMap ablations")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["shared_coefficients", "linear_mapper",
                                 "top_layers_only", "middle_layers_only",
                                 "bottom_layers_only"])
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--active_fraction", type=float, default=0.75)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_eval_examples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class LinearMapper(nn.Module):
    """Single linear layer replacing the modulated mapper (Exp 6)."""
    def __init__(self, code_dim, num_blocks, rank):
        super().__init__()
        output_dim = num_blocks + 3 * num_blocks * rank
        self.linear = nn.Linear(code_dim, output_dim)
        self.num_blocks = num_blocks
        self.rank = rank

    def forward(self, layer_idx, z):
        out = self.linear(z)
        q = out[:self.num_blocks]
        offset = self.num_blocks
        c_u = out[offset:offset + self.num_blocks * self.rank].reshape(self.num_blocks, self.rank)
        offset += self.num_blocks * self.rank
        c_g = out[offset:offset + self.num_blocks * self.rank].reshape(self.num_blocks, self.rank)
        offset += self.num_blocks * self.rank
        c_d = out[offset:].reshape(self.num_blocks, self.rank)
        return q, c_u, c_g, c_d


def train_ablation(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.output_dir is None:
        args.output_dir = f"outputs/ablation_{args.ablation}"
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

    tm_config = TaskMapConfig.from_backbone(
        args.backbone, block_size=128, active_fraction=args.active_fraction,
        code_dim=32, rank=8, mapper_hidden=512, total_steps=args.max_steps,
    )

    taskmap = TaskMapModel(tm_config, num_tasks=len(task_ids), freeze_mapper=False).to(device)
    taskmap.register_tasks(task_ids)

    for tid in task_ids:
        desc = KNOWN_TASKS[tid]["descriptions"][0]
        embed = taskmap.task_code.compute_description_embedding(
            backbone, tokenizer, desc, device
        )
        taskmap.cache_description(tid, embed)

    hook_manager = TaskMapHookManager(backbone, taskmap, block_size=128)

    # ── Ablation-specific modifications ──
    if args.ablation == "linear_mapper":
        print("\n=== Ablation: Linear mapper (replacing modulated mapper) ===")
        linear_map = LinearMapper(tm_config.code_dim, tm_config.num_blocks, tm_config.rank).to(device)
        original_mapper = taskmap.mapper_bank
        taskmap.mapper_bank = linear_map

    elif args.ablation == "shared_coefficients":
        print("\n=== Ablation: Shared coefficients (one per layer, not per block) ===")

    elif args.ablation in ("top_layers_only", "middle_layers_only", "bottom_layers_only"):
        n = tm_config.num_layers
        if args.ablation == "top_layers_only":
            active_layers = list(range(n * 2 // 3, n))
        elif args.ablation == "middle_layers_only":
            active_layers = list(range(n // 3, n * 2 // 3))
        else:
            active_layers = list(range(0, n // 3))
        print(f"\n=== Ablation: {args.ablation} — active layers: {active_layers} ===")

    # ── Optimizer ──
    param_groups = []
    code_params = [p for n, p in taskmap.task_code.named_parameters()
                   if p.requires_grad and "projector" not in n]
    proj_params = [p for n, p in taskmap.task_code.named_parameters()
                   if p.requires_grad and "projector" in n]
    param_groups.append({"params": code_params, "lr": 2e-3})
    param_groups.append({"params": proj_params, "lr": 2e-4})

    if args.ablation == "linear_mapper":
        param_groups.append({"params": linear_map.parameters(), "lr": 2e-4})
    else:
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

    # ── Training loop ──
    grad_accum = args.gradient_accumulation_steps
    print(f"\nStarting ablation '{args.ablation}' for {max_steps} steps...")

    global_step = 0
    accum_loss = 0.0
    all_route_masks = {}
    t_start = time.time()

    dataloader = build_dataloader(train_data, args.microbatch_size,
                                  max_steps * grad_accum, args.seed)

    for step_idx, (task_id, examples) in enumerate(dataloader):
        taskmap.clear_route_cache()
        routes = taskmap.compute_route(task_id, device)

        if args.ablation == "shared_coefficients":
            # Use same coefficient for all blocks in each layer
            for r in routes:
                mean_cu = r['c_u'].mean(dim=0, keepdim=True).expand_as(r['c_u'])
                mean_cg = r['c_g'].mean(dim=0, keepdim=True).expand_as(r['c_g'])
                mean_cd = r['c_d'].mean(dim=0, keepdim=True).expand_as(r['c_d'])
                r['c_u'] = mean_cu
                r['c_g'] = mean_cg
                r['c_d'] = mean_cd

        masks = [r['mask'].detach() for r in routes]

        if args.ablation in ("top_layers_only", "middle_layers_only", "bottom_layers_only"):
            # Zero out masks for inactive layers
            for l in range(len(masks)):
                if l not in active_layers:
                    masks[l] = torch.zeros_like(masks[l])
                    routes[l]['mask'] = masks[l]
                    routes[l]['selected'] = []

        all_route_masks[task_id] = masks
        hook_manager.activate_for_task(task_id, device)

        batch = tokenize_batch(tokenizer, examples, args.max_seq_length)
        batch = {k_: v.to(device) for k_, v in batch.items()}

        outputs = backbone(**batch)
        task_loss = outputs.loss

        is_warmup = taskmap.router.is_warmup()
        total_loss, loss_dict = loss_computer.compute(
            task_loss, taskmap, task_id, all_route_masks, is_warmup
        )

        scaled_loss = total_loss / grad_accum
        scaled_loss.backward()
        accum_loss += scaled_loss.item()

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

    # ── Evaluate ──
    print(f"\n=== Evaluating {args.ablation} ===")
    eval_data = format_all_tasks(datasets, split="validation")
    for tid in eval_data:
        if len(eval_data[tid]) > args.max_eval_examples:
            eval_data[tid] = eval_data[tid][:args.max_eval_examples]

    backbone.eval()
    all_scores = {}
    for tid, examples in eval_data.items():
        if tid not in KNOWN_TASKS:
            continue

        taskmap.clear_route_cache()
        routes = taskmap.compute_route(tid, device)

        if args.ablation == "shared_coefficients":
            for r in routes:
                mean_cu = r['c_u'].mean(dim=0, keepdim=True).expand_as(r['c_u'])
                mean_cg = r['c_g'].mean(dim=0, keepdim=True).expand_as(r['c_g'])
                mean_cd = r['c_d'].mean(dim=0, keepdim=True).expand_as(r['c_d'])
                r['c_u'] = mean_cu
                r['c_g'] = mean_cg
                r['c_d'] = mean_cd

        if args.ablation in ("top_layers_only", "middle_layers_only", "bottom_layers_only"):
            for l in range(len(routes)):
                if l not in active_layers:
                    routes[l]['mask'] = torch.zeros_like(routes[l]['mask'])
                    routes[l]['selected'] = []

        hook_manager.activate_for_task(tid, device)
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
    print(f"\n  {args.ablation} macro: {macro:.2f}")

    results = {
        "mode": f"ablation_{args.ablation}",
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
    train_ablation(args)
