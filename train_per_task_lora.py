"""
Per-task LoRA training: one separate LoRA adapter per task.

Trains a fresh LoRA adapter for each task independently, then evaluates
each on its own validation set. This is the strongest LoRA baseline
(no cross-task interference by construction).

Usage:
  python train_per_task_lora.py --backbone Qwen/Qwen2.5-1.5B --steps_per_task 2000
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

from models.backbone import load_backbone, add_lora, count_parameters
from data.config import KNOWN_TASKS
from data.download import download_task
from data.format import format_dataset, format_all_tasks
from data.sampler import build_dataloader
from train import tokenize_batch, set_seed
from eval import evaluate_task


def parse_args():
    parser = argparse.ArgumentParser(description="Per-task LoRA Training")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--steps_per_task", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_eval_examples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="outputs/per_task_lora")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_per_task_lora(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading backbone: {args.backbone}")
    datasets = {}
    for tid, meta in KNOWN_TASKS.items():
        ds = download_task(tid, meta)
        if ds is not None:
            datasets[tid] = ds

    train_data = format_all_tasks(datasets, split="train")
    eval_data = format_all_tasks(datasets, split="validation")
    task_ids = list(train_data.keys())
    print(f"Tasks: {task_ids}")

    for tid in eval_data:
        if len(eval_data[tid]) > args.max_eval_examples:
            eval_data[tid] = eval_data[tid][:args.max_eval_examples]

    all_scores = {}

    for tid in task_ids:
        if tid not in KNOWN_TASKS:
            continue
        print(f"\n{'='*60}")
        print(f"Training LoRA for task: {tid}")
        print(f"{'='*60}")

        set_seed(args.seed)
        backbone, tokenizer = load_backbone(args.backbone)
        backbone = backbone.to(device)
        backbone = add_lora(backbone, rank=args.lora_rank, alpha=args.lora_alpha)
        backbone.train()

        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        optimizer = AdamW(
            [p for p in backbone.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        max_steps = args.steps_per_task
        warmup_steps = int(max_steps * 0.03)
        warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1))
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(max_steps - warmup_steps, 1))
        scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

        single_task_data = {tid: train_data[tid]}
        grad_accum = args.gradient_accumulation_steps
        dataloader = build_dataloader(single_task_data, args.microbatch_size,
                                      max_steps * grad_accum, args.seed)

        global_step = 0
        accum_loss = 0.0
        t_start = time.time()

        for step_idx, (_, examples) in enumerate(dataloader):
            batch = tokenize_batch(tokenizer, examples, args.max_seq_length)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = backbone(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in backbone.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 200 == 0 or global_step == 1:
                    elapsed = time.time() - t_start
                    print(f"  Step {global_step}/{max_steps} | "
                          f"Loss: {accum_loss:.4f} | Task: {tid} | Time: {elapsed:.0f}s")
                    accum_loss = 0.0

                if global_step >= max_steps:
                    break

        print(f"  Training complete for {tid} in {time.time() - t_start:.0f}s")

        # Evaluate this task
        backbone.eval()
        if tid in eval_data and tid in KNOWN_TASKS:
            scores = evaluate_task(
                backbone, tokenizer, tid, eval_data[tid],
                KNOWN_TASKS[tid]["metric"], KNOWN_TASKS[tid]["max_response_tokens"], device
            )
            all_scores[tid] = scores
            print(f"  {tid}: {scores}")

        # Free memory before next task
        del backbone, tokenizer, optimizer, scheduler
        torch.cuda.empty_cache()

    # Compute macro
    primary_scores = [list(v.values())[0] for v in all_scores.values() if isinstance(v, dict)]
    # Exclude MBPP from macro
    eval_tasks = [t for t in all_scores if t != 'mbpp' and isinstance(all_scores[t], dict)]
    primary_no_mbpp = [list(all_scores[t].values())[0] for t in eval_tasks]
    macro = np.mean(primary_no_mbpp) if primary_no_mbpp else 0.0
    all_scores["macro_avg_9task"] = float(macro)
    print(f"\n  Per-task LoRA macro (excl MBPP): {macro:.2f}")

    results = {
        "mode": "per_task_lora",
        "backbone": args.backbone,
        "lora_rank": args.lora_rank,
        "steps_per_task": args.steps_per_task,
        "eval_examples": args.max_eval_examples,
        "scores": all_scores,
    }
    print("\n=== RESULTS JSON ===")
    print(json.dumps(results, indent=2, default=str))
    print("=== END RESULTS ===")

    output_file = os.path.join(args.output_dir, "eval_per_task_lora.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    train_per_task_lora(args)
