"""
Cold-start baseline evaluation.

Evaluates held-out tasks under multiple conditions to establish
whether description-conditioned routing improves over baselines.

Baselines:
1. Frozen backbone (no routing, no adaptation)
2. Random route (random block selection at same ρ)
3. Same-family route (use route from nearest known task in same family)

Usage:
  python eval_coldstart_baselines.py --backbone Qwen/Qwen2.5-1.5B \
    --taskmap_checkpoint outputs/taskmap/final/taskmap_state.pt
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone
from models.taskmap_model import TaskMapModel, TaskMapConfig
from models.ffn_hooks import TaskMapHookManager
from data.config import KNOWN_TASKS, COLD_START_TASKS, FAMILY_PAIRS
from data.download import download_task
from data.format import format_dataset
from eval import evaluate_task
from train import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Cold-start baseline evaluation")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--taskmap_checkpoint", type=str, default=None)
    parser.add_argument("--active_fraction", type=float, default=0.75)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="outputs/coldstart_baselines.json")
    return parser.parse_args()


def eval_coldstart_baselines(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print(f"Loading backbone: {args.backbone}")
    backbone, tokenizer = load_backbone(args.backbone)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Load cold-start task data
    cs_data = {}
    for cs_tid, cs_meta in COLD_START_TASKS.items():
        try:
            ds = download_task(cs_tid, cs_meta)
            if ds is None:
                continue
            split_name = cs_meta["split_map"].get("test")
            if split_name not in ds:
                continue
            formatted = format_dataset(cs_tid, ds[split_name], "validation")
            if len(formatted) > args.max_eval:
                formatted = formatted[:args.max_eval]
            cs_data[cs_tid] = (formatted, cs_meta)
            print(f"  Loaded {cs_tid}: {len(formatted)} examples")
        except Exception as e:
            print(f"  Skipping {cs_tid}: {e}")

    if not cs_data:
        print("No cold-start tasks loaded")
        return

    results = {}

    # === Baseline 1: Frozen backbone (no routing) ===
    print("\n=== Baseline 1: Frozen backbone ===")
    frozen_scores = {}
    for cs_tid, (examples, meta) in cs_data.items():
        try:
            scores = evaluate_task(
                backbone, tokenizer, cs_tid, examples,
                meta["metric"], meta["max_response_tokens"], device
            )
            frozen_scores[cs_tid] = scores
            print(f"  {cs_tid}: {scores}")
        except Exception as e:
            print(f"  {cs_tid} failed: {e}")
    results["frozen"] = frozen_scores

    # === Setup TaskMap for routing baselines ===
    if args.taskmap_checkpoint:
        print(f"\nLoading TaskMap checkpoint: {args.taskmap_checkpoint}")
        checkpoint = torch.load(args.taskmap_checkpoint, map_location=device)
        cfg = checkpoint.get("config", {})

        task_ids = list(KNOWN_TASKS.keys())
        tm_config = TaskMapConfig.from_backbone(
            args.backbone,
            block_size=cfg.get("block_size", args.block_size),
            active_fraction=cfg.get("active_fraction", args.active_fraction),
            code_dim=cfg.get("code_dim", 32),
            rank=cfg.get("rank", 8),
            mapper_hidden=cfg.get("mapper_hidden", 512),
        )

        taskmap = TaskMapModel(tm_config, num_tasks=len(task_ids), freeze_mapper=False).to(device)
        taskmap.register_tasks(task_ids)

        if "task_code_state" in checkpoint:
            taskmap.task_code.load_state_dict(checkpoint["task_code_state"], strict=False)
        if "mapper_state" in checkpoint:
            taskmap.mapper_bank.load_state_dict(checkpoint["mapper_state"], strict=False)

        # Compute description embeddings for known tasks
        for tid in task_ids:
            desc = KNOWN_TASKS[tid]["descriptions"][0]
            embed = taskmap.task_code.compute_description_embedding(
                backbone, tokenizer, desc, device
            )
            taskmap.cache_description(tid, embed)

        hook_manager = TaskMapHookManager(
            backbone, taskmap, block_size=cfg.get("block_size", args.block_size)
        )

        # === Baseline 2: Random route ===
        print("\n=== Baseline 2: Random route ===")
        random_scores = {}
        k = max(1, int(args.active_fraction * tm_config.num_blocks))
        for cs_tid, (examples, meta) in cs_data.items():
            try:
                # Generate random route and activate
                for layer_idx, hook in enumerate(hook_manager.hooks):
                    if layer_idx >= tm_config.num_layers:
                        break
                    # Random block selection
                    perm = torch.randperm(tm_config.num_blocks)[:k]
                    mask = torch.zeros(tm_config.num_blocks, device=device)
                    mask[perm] = 1.0
                    # Zero coefficients (no residual modification)
                    route_info = {
                        'mask': mask,
                        'selected': perm.tolist(),
                        'c_u': torch.zeros(tm_config.num_blocks, tm_config.rank, device=device),
                        'c_g': torch.zeros(tm_config.num_blocks, tm_config.rank, device=device),
                        'c_d': torch.zeros(tm_config.num_blocks, tm_config.rank, device=device),
                    }
                    hook.set_route(route_info, taskmap.residual_bases)

                scores = evaluate_task(
                    backbone, tokenizer, cs_tid, examples,
                    meta["metric"], meta["max_response_tokens"], device
                )
                random_scores[cs_tid] = scores
                print(f"  {cs_tid}: {scores}")
            except Exception as e:
                print(f"  {cs_tid} failed: {e}")
        results["random_route"] = random_scores

        # === Baseline 3: Same-family route ===
        print("\n=== Baseline 3: Same-family route (nearest known task) ===")
        family_scores = {}
        for cs_tid, (examples, meta) in cs_data.items():
            cs_family = meta["family"]
            # Find a known task from the same family
            same_family_tasks = [t for t in task_ids if KNOWN_TASKS[t]["family"] == cs_family]
            if not same_family_tasks:
                print(f"  {cs_tid}: no same-family known task")
                continue
            donor_tid = same_family_tasks[0]
            print(f"  {cs_tid} (family={cs_family}): using route from {donor_tid}")
            try:
                taskmap.clear_route_cache()
                hook_manager.activate_for_task(donor_tid, device)
                scores = evaluate_task(
                    backbone, tokenizer, cs_tid, examples,
                    meta["metric"], meta["max_response_tokens"], device
                )
                family_scores[cs_tid] = scores
                print(f"  {cs_tid}: {scores}")
            except Exception as e:
                print(f"  {cs_tid} failed: {e}")
        results["same_family_route"] = family_scores

        # === TaskMap description-conditioned route ===
        print("\n=== TaskMap: description-conditioned route ===")
        desc_scores = {}
        for cs_tid, (examples, meta) in cs_data.items():
            try:
                cs_desc = meta["descriptions"][0]
                cs_embed = taskmap.task_code.compute_description_embedding(
                    backbone, tokenizer, cs_desc, device
                )
                taskmap.cache_description(cs_tid, cs_embed)
                taskmap.clear_route_cache()
                taskmap.compute_route(cs_tid, device)
                hook_manager.activate_for_task(cs_tid, device)
                scores = evaluate_task(
                    backbone, tokenizer, cs_tid, examples,
                    meta["metric"], meta["max_response_tokens"], device
                )
                desc_scores[cs_tid] = scores
                print(f"  {cs_tid}: {scores}")
            except Exception as e:
                print(f"  {cs_tid} failed: {e}")
        results["description_route"] = desc_scores

        hook_manager.remove_all()

    # Print summary
    print("\n=== COLD-START BASELINES SUMMARY ===")
    print(json.dumps(results, indent=2, default=str))
    print("=== END ===")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    args = parse_args()
    eval_coldstart_baselines(args)
