"""
Two-Phase Alternating Training for TaskMap.

Novel approach: decouple task-code optimization from backbone computation.

Phase 1 (Code Optimization — no backbone needed):
  Optimize task codes using only mapping losses (stability, smoothness, alignment)
  through the frozen mapper. This is pure small-network optimization on 12K params.
  Runs fast because it only involves the mapper forward pass, not the 1.5B backbone.

Phase 2 (Backbone Validation — GPU):
  Every N code-optimization steps, run the backbone with hooks to compute actual
  task loss and update code gradients with real signal.

The mapping losses enforce that nearby codes produce similar routes/coefficients,
so optimization in latent space transfers to the target weight space (Mapping
Networks theorem).

Usage:
  python train_twophase.py --backbone Qwen/Qwen2.5-1.5B --max_steps 6000
  python train_twophase.py --backbone Qwen/Qwen2.5-1.5B --code_steps 50 --backbone_every 10
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone, count_parameters
from models.taskmap_model import TaskMapModel, TaskMapConfig
from models.ffn_hooks import TaskMapHookManager
from data.config import KNOWN_TASKS, FAMILY_PAIRS, COLD_START_TASKS
from data.download import download_task
from data.format import format_all_tasks, format_dataset
from data.sampler import build_dataloader
from train import tokenize_batch, set_seed
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Two-Phase Alternating TaskMap Training")
    parser.add_argument("--config", type=str, default="configs/taskmap_reference.yaml")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--code_steps", type=int, default=50,
                        help="Code-only optimization steps between backbone validations")
    parser.add_argument("--backbone_every", type=int, default=10,
                        help="Run backbone validation every N code-optimization rounds")
    parser.add_argument("--code_lr", type=float, default=5e-3)
    parser.add_argument("--backbone_lr", type=float, default=2e-4)
    parser.add_argument("--active_fraction", type=float, default=0.75)
    parser.add_argument("--code_dim", type=int, default=8)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="outputs/twophase")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def mapping_loss_only(taskmap, task_ids, device, config):
    """
    Compute mapping losses (stability + alignment) without backbone.
    Only uses the mapper, which is a small network.
    Returns differentiable loss for task code optimization.
    """
    import random
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n = 0

    for tid in random.sample(task_ids, min(3, len(task_ids))):
        l_idx = random.randint(0, config.num_layers - 1)
        z = taskmap.task_code.get_code(tid, l_idx, device)
        mapper_fn = lambda z_in, l=l_idx: taskmap.mapper_bank(l, z_in)

        # Stability: mapper output should be stable under perturbation
        delta = torch.randn_like(z) * 0.01
        out_clean = mapper_fn(z)
        out_noisy = mapper_fn(z + delta)
        clean_cat = torch.cat([o.flatten() for o in out_clean])
        noisy_cat = torch.cat([o.flatten() for o in out_noisy])
        stab = (noisy_cat - clean_cat).pow(2).sum() / (delta.pow(2).sum() + 1e-8)

        # Smoothness: small Jacobian norm
        v = torch.randn_like(z)
        z_v = z.detach().requires_grad_(True)
        out_v = mapper_fn(z_v)
        out_v_cat = torch.cat([o.flatten() for o in out_v])
        jvp = torch.autograd.grad(out_v_cat.sum(), z_v, create_graph=True)[0]
        smooth = (jvp * v).pow(2).sum()

        # Alignment: code direction should align with mapper output direction
        output_dim = config.num_blocks + 3 * config.num_blocks * config.rank
        R = torch.randn(config.code_dim, output_dim, device=device) * (1.0 / math.sqrt(output_dim))
        o_cat = torch.cat([o.flatten() for o in out_clean])
        projected = R @ o_cat
        z_norm = F.normalize(z.unsqueeze(0), dim=-1)
        p_norm = F.normalize(projected.unsqueeze(0), dim=-1)
        align = 1.0 - (z_norm * p_norm).sum()

        total_loss = total_loss + 0.1 * stab + 0.01 * smooth + 0.01 * align
        n += 1

    return total_loss / max(n, 1)


def train_twophase(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load backbone
    print(f"Loading frozen backbone: {args.backbone}")
    backbone, tokenizer = load_backbone(args.backbone)
    backbone = backbone.to(device).eval()
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
    print(f"Tasks: {task_ids}, Total: {sum(len(v) for v in train_data.values()):,}")

    # Setup TaskMap (frozen mapper, compact codes)
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    config = TaskMapConfig.from_backbone(
        args.backbone,
        block_size=cfg.get("block_size", 128),
        active_fraction=args.active_fraction,
        code_dim=args.code_dim,
        rank=cfg.get("rank", 8),
        mapper_hidden=cfg.get("mapper_hidden", 512),
        total_steps=args.max_steps,
    )

    taskmap = TaskMapModel(config, num_tasks=len(task_ids),
                           freeze_mapper=True,
                           shared_projector=True,
                           global_code=True).to(device)
    taskmap.register_tasks(task_ids)

    # Cache description embeddings
    for tid in task_ids:
        meta = KNOWN_TASKS[tid]
        embed = taskmap.task_code.compute_description_embedding(
            backbone, tokenizer, meta["descriptions"][0], device
        )
        taskmap.cache_description(tid, embed)

    # Install hooks
    hook_manager = TaskMapHookManager(backbone, taskmap, block_size=cfg.get("block_size", 128))

    trainable_params = taskmap.trainable_parameters()
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"\nTrainable params: {total_trainable:,}")
    print(f"vs LoRA r=8: {7_100_000 / total_trainable:.0f}x fewer")

    # Two separate optimizers
    code_optimizer = AdamW(trainable_params, lr=args.code_lr, weight_decay=0.01)
    backbone_optimizer = AdamW(trainable_params, lr=args.backbone_lr, weight_decay=0.01)

    max_steps = 2 if args.dry_run else args.max_steps
    code_steps = args.code_steps
    backbone_every = args.backbone_every

    # Training loop
    print(f"\n=== Two-Phase Alternating Training ===")
    print(f"  Code optimization steps per round: {code_steps}")
    print(f"  Backbone validation every: {backbone_every} rounds")
    print(f"  Total backbone steps: ~{max_steps}")

    global_step = 0
    round_num = 0
    t_start = time.time()
    code_time = 0
    backbone_time = 0

    dataloader_iter = iter(build_dataloader(
        train_data, args.microbatch_size,
        max_steps * args.gradient_accumulation_steps * 2, args.seed
    ))

    while global_step < max_steps:
        round_num += 1

        # ── Phase 1: Code optimization (no backbone) ──
        t_code_start = time.time()
        code_loss_avg = 0
        for cs in range(code_steps):
            code_optimizer.zero_grad()
            loss = mapping_loss_only(taskmap, task_ids, device, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            code_optimizer.step()
            code_loss_avg += loss.item()
        code_loss_avg /= code_steps
        code_time += time.time() - t_code_start

        # ── Phase 2: Backbone validation (with hooks) ──
        if round_num % backbone_every == 0 or round_num == 1:
            t_bb_start = time.time()
            bb_loss_avg = 0
            bb_steps = 0

            for _ in range(args.gradient_accumulation_steps):
                try:
                    task_id, examples = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(build_dataloader(
                        train_data, args.microbatch_size,
                        max_steps * args.gradient_accumulation_steps, args.seed
                    ))
                    task_id, examples = next(dataloader_iter)

                taskmap.clear_route_cache()
                hook_manager.activate_for_task(task_id, device)

                batch = tokenize_batch(tokenizer, examples, args.max_seq_length)
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = backbone(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                bb_loss_avg += loss.item()
                bb_steps += 1

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            backbone_optimizer.step()
            backbone_optimizer.zero_grad()
            global_step += 1
            backbone_time += time.time() - t_bb_start

            if global_step % 100 == 0 or global_step == 1:
                elapsed = time.time() - t_start
                print(f"  Step {global_step}/{max_steps} | "
                      f"BB loss: {bb_loss_avg:.4f} | Code loss: {code_loss_avg:.4f} | "
                      f"Round: {round_num} | "
                      f"Code time: {code_time:.0f}s | BB time: {backbone_time:.0f}s | "
                      f"Total: {elapsed:.0f}s")

            if global_step >= max_steps:
                break
        else:
            # Count code-only rounds as partial steps for progress
            global_step += 1
            if global_step % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  Step {global_step}/{max_steps} | "
                      f"Code loss: {code_loss_avg:.4f} | "
                      f"Round: {round_num} (code-only) | "
                      f"Code time: {code_time:.0f}s | BB time: {backbone_time:.0f}s | "
                      f"Total: {elapsed:.0f}s")

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({global_step} steps)")
    print(f"  Code optimization: {code_time:.0f}s ({code_time/total_time*100:.0f}%)")
    print(f"  Backbone forward:  {backbone_time:.0f}s ({backbone_time/total_time*100:.0f}%)")

    # ── Evaluate ──
    print("\n=== Evaluation ===")
    eval_datasets = {}
    for tid, meta in KNOWN_TASKS.items():
        ds = download_task(tid, meta)
        if ds is not None:
            eval_datasets[tid] = ds
    eval_data = format_all_tasks(eval_datasets, split="validation")
    max_eval = 50
    for tid in eval_data:
        if len(eval_data[tid]) > max_eval:
            eval_data[tid] = eval_data[tid][:max_eval]

    backbone.eval()
    from eval import evaluate_task
    all_scores = {}
    for tid, examples in eval_data.items():
        if tid not in KNOWN_TASKS:
            continue
        hook_manager.activate_for_task(tid, device)
        scores = evaluate_task(
            backbone, tokenizer, tid, examples,
            KNOWN_TASKS[tid]["metric"], KNOWN_TASKS[tid]["max_response_tokens"], device
        )
        all_scores[tid] = scores
        print(f"    {tid}: {scores}")

    primary_scores = [list(v.values())[0] for v in all_scores.values()]
    macro = np.mean(primary_scores) if primary_scores else 0.0
    all_scores["macro_avg"] = macro
    print(f"\n  Macro average: {macro:.2f}")

    # Results
    results = {
        "mode": "twophase",
        "trainable_params": total_trainable,
        "scores": all_scores,
        "timing": {
            "total_s": total_time,
            "code_optimization_s": code_time,
            "backbone_forward_s": backbone_time,
            "code_pct": code_time / total_time * 100,
            "backbone_pct": backbone_time / total_time * 100,
        }
    }
    print("\n=== RESULTS JSON ===")
    print(json.dumps(results, indent=2, default=str))
    print("=== END RESULTS ===")

    hook_manager.remove_all()


if __name__ == "__main__":
    args = parse_args()
    train_twophase(args)
