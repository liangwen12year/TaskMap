"""
TaskMap training script (Section 3.7, Algorithm 1).

Key differences from baseline train.py:
1. Task-homogeneous microbatches (route computed once per task per step)
2. Only task codes (P_l, r_{t,l}) receive gradients; backbone + mapper frozen
3. Separate learning rates for codes vs projectors
4. 7 loss terms with schedule-dependent activation
5. Route cache cleared once per optimizer step

Usage:
  python train_taskmap.py --config configs/taskmap_reference.yaml
  python train_taskmap.py --dry_run  # 2 steps on CPU
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone, count_parameters
from models.taskmap_model import TaskMapModel, TaskMapConfig
from models.ffn_hooks import TaskMapHookManager
from data.config import KNOWN_TASKS, FAMILY_PAIRS, FAMILY_TO_TASKS
from data.download import download_task
from data.format import format_all_tasks
from data.sampler import build_dataloader
from losses import TaskMapLossComputer
from train import tokenize_batch, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="TaskMap Training")
    parser.add_argument("--config", type=str, default="configs/taskmap_reference.yaml")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_config(args):
    """Load YAML config and apply CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    return cfg


def setup_taskmap(cfg, backbone_model, tokenizer, task_ids, device):
    """Initialize TaskMap components."""
    tm_config = TaskMapConfig.from_backbone(
        cfg["backbone"],
        block_size=cfg.get("block_size", 128),
        active_fraction=cfg.get("active_fraction", 0.50),
        code_dim=cfg.get("code_dim", 32),
        rank=cfg.get("rank", 8),
        mapper_hidden=cfg.get("mapper_hidden", 512),
        total_steps=cfg.get("max_steps", 12000),
        warmup_fraction=cfg.get("warmup_fraction", 0.03),
    )

    taskmap = TaskMapModel(tm_config, num_tasks=len(task_ids)).to(device)
    taskmap.register_tasks(task_ids)

    # Compute and cache description embeddings for all tasks
    print("Computing description embeddings...")
    for tid in task_ids:
        meta = KNOWN_TASKS[tid]
        description = meta["descriptions"][0]
        embed = taskmap.task_code.compute_description_embedding(
            backbone_model, tokenizer, description, device
        )
        taskmap.cache_description(tid, embed)
        print(f"  {tid}: '{description[:50]}...' -> embed norm={embed.norm():.3f}")

    # Install FFN hooks
    hook_manager = TaskMapHookManager(
        backbone_model, taskmap, block_size=cfg.get("block_size", 128)
    )

    return taskmap, tm_config, hook_manager


def train_taskmap(args):
    cfg = load_config(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.get("output_dir", "outputs/taskmap"), exist_ok=True)

    # ── Load backbone (frozen) ──
    backbone_name = cfg.get("backbone", "Qwen/Qwen2.5-1.5B")
    print(f"Loading frozen backbone: {backbone_name}")
    backbone_model, tokenizer = load_backbone(backbone_name)
    backbone_model = backbone_model.to(device)
    backbone_model.eval()
    for p in backbone_model.parameters():
        p.requires_grad = False

    # ── Load data ──
    print("\nLoading training data...")
    datasets = {}
    for task_id, meta in KNOWN_TASKS.items():
        ds = download_task(task_id, meta)
        if ds is not None:
            datasets[task_id] = ds

    train_data = format_all_tasks(datasets, split="train")
    task_ids = list(train_data.keys())
    total_examples = sum(len(v) for v in train_data.values())
    print(f"Tasks loaded: {task_ids}")
    print(f"Total training examples: {total_examples:,}")

    # ── Setup TaskMap ──
    taskmap, tm_config, hook_manager = setup_taskmap(cfg, backbone_model, tokenizer, task_ids, device)
    summary = taskmap.parameter_summary()
    print(f"\nTaskMap parameter summary: {summary}")

    # ── Optimizer (separate param groups) ──
    code_params = []
    projector_params = []
    for name, param in taskmap.task_code.named_parameters():
        if param.requires_grad:
            if "projector" in name:
                projector_params.append(param)
            else:
                code_params.append(param)

    optimizer = AdamW([
        {"params": code_params, "lr": cfg.get("code_learning_rate", 2e-3)},
        {"params": projector_params, "lr": cfg.get("projector_learning_rate", 2e-4)},
    ], weight_decay=cfg.get("weight_decay", 0.01), betas=(0.9, 0.95))

    max_steps = 2 if args.dry_run else cfg.get("max_steps", 12000)
    warmup_steps = int(max_steps * cfg.get("warmup_fraction", 0.03))
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(max_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    # ── Loss computer ──
    task_families = {tid: KNOWN_TASKS[tid]["family"] for tid in task_ids}
    loss_computer = TaskMapLossComputer(
        tm_config, FAMILY_PAIRS, task_families,
        lambda_bud=cfg.get("lambda_bud", 0.05),
        lambda_topo=cfg.get("lambda_topo", 0.01),
        lambda_bal=cfg.get("lambda_bal", 0.01),
        lambda_stab=cfg.get("lambda_stab", 1e-3),
        lambda_sm=cfg.get("lambda_sm", 1e-3),
        lambda_align=cfg.get("lambda_align", 1e-4),
    )

    # ── Training loop ──
    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    max_seq = cfg.get("max_seq_length", 2048)
    microbatch_size = cfg.get("microbatch_size", 4)

    print(f"\nStarting TaskMap training for {max_steps} steps...")
    print(f"  Warmup: {warmup_steps} steps (dense), then Gumbel anneal")
    print(f"  Active fraction: {tm_config.active_fraction} ({taskmap.router.k}/{tm_config.num_blocks} blocks)")

    global_step = 0
    accum_loss = 0.0
    all_route_masks = {}
    t_start = time.time()

    dataloader = build_dataloader(train_data, microbatch_size,
                                  max_steps * grad_accum, args.seed)

    for step_idx, (task_id, examples) in enumerate(dataloader):
        # ── Compute route fresh each microbatch (no caching) ──
        # Must recompute to get fresh graph nodes for backward
        taskmap.clear_route_cache()
        routes = taskmap.compute_route(task_id, device)
        masks_for_task = [r['mask'].detach() for r in routes]
        all_route_masks[task_id] = masks_for_task
        hook_manager.activate_for_task(task_id, device)

        # ── Forward: backbone with task-conditioned FFN hooks ──
        batch = tokenize_batch(tokenizer, examples, max_seq)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = backbone_model(**batch)
        task_loss = outputs.loss

        # ── Compute TaskMap losses ──
        is_warmup = taskmap.router.is_warmup()
        total_loss, loss_dict = loss_computer.compute(
            task_loss, taskmap, task_id, all_route_masks, is_warmup
        )

        scaled_loss = total_loss / grad_accum
        scaled_loss.backward()
        accum_loss += scaled_loss.item()

        if (step_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                taskmap.trainable_parameters(), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            taskmap.step()
            global_step += 1

            if global_step % 100 == 0 or global_step == 1:
                elapsed = time.time() - t_start
                temp = taskmap.router.get_temperature()
                phase = "warmup" if is_warmup else f"temp={temp:.3f}"
                loss_str = " | ".join(f"{k}={v.item():.4f}" if torch.is_tensor(v) else f"{k}={v:.4f}"
                                       for k, v in loss_dict.items() if k != "total")
                print(f"  Step {global_step}/{max_steps} | "
                      f"Total: {accum_loss:.4f} | {loss_str} | "
                      f"Task: {task_id} | Phase: {phase} | "
                      f"Time: {elapsed:.0f}s")
                accum_loss = 0.0

            if not args.dry_run and global_step % cfg.get("save_every", 2000) == 0:
                save_path = os.path.join(cfg["output_dir"], f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    "task_code_state": taskmap.task_code.state_dict(),
                    "step": global_step,
                    "config": cfg,
                }, os.path.join(save_path, "taskmap_state.pt"))
                print(f"  Saved checkpoint to {save_path}")

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({global_step} steps)")

    # ── Save final ──
    final_path = os.path.join(cfg.get("output_dir", "outputs/taskmap"), "final")
    os.makedirs(final_path, exist_ok=True)
    torch.save({
        "task_code_state": taskmap.task_code.state_dict(),
        "mapper_state": taskmap.mapper_bank.state_dict(),
        "config": cfg,
        "step": global_step,
    }, os.path.join(final_path, "taskmap_state.pt"))
    print(f"Saved final model to {final_path}")

    hook_manager.remove_all()
    return taskmap, backbone_model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    train_taskmap(args)
