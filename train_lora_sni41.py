"""
Shared multi-task LoRA baseline on 41 SNI training tasks.

Trains a single shared LoRA adapter on the same 41 Super-NaturalInstructions
tasks used by TaskMap, then evaluates unchanged on 20 held-out tasks.
This provides a cold-start baseline: the shared LoRA has never seen the
held-out tasks, but its shared adaptation may generalize.

Usage:
  python train_lora_sni41.py --backbone Qwen/Qwen2.5-1.5B --max_steps 6000
  python train_lora_sni41.py --dry_run  # 2 steps on CPU
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone, add_lora, count_parameters
from data.task_collection import (
    TRAIN_TASKS_SNI,
    HOLDOUT_TASKS_SNI,
    load_sni_dataset,
    filter_sni_tasks,
    format_sni_examples,
)
from data.sampler import build_dataloader
from train import tokenize_batch, set_seed
from eval import METRIC_FNS, generate_predictions, accuracy, rouge_l
from train_taskmap_scaled import (
    infer_metric,
    infer_family_from_definition,
    validate_sni_tasks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shared LoRA baseline on 41 SNI tasks")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_per_task", type=int, default=2000)
    parser.add_argument("--max_eval_examples", type=int, default=200)
    parser.add_argument("--sni_cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/lora_sni41")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_sni_data(task_list, split, max_per_task, cache_dir=None):
    """Load and format SNI data for the requested tasks."""
    full_ds = load_sni_dataset(cache_dir)
    valid_tasks, missing = validate_sni_tasks(full_ds, task_list)
    if missing:
        print(f"  WARNING: {len(missing)} tasks not found: {missing[:5]}...")

    raw_data = filter_sni_tasks(full_ds, valid_tasks, max_per_task)

    task_data = {}
    task_definitions = {}
    task_families = {}
    task_metrics = {}

    for task_name in valid_tasks:
        if task_name not in raw_data or not raw_data[task_name]:
            continue
        definition = raw_data[task_name][0].get("definition", "")
        task_definitions[task_name] = definition
        task_families[task_name] = infer_family_from_definition(
            task_name, definition)

        formatted = format_sni_examples(task_name, raw_data[task_name], split)
        if not formatted:
            continue
        task_data[task_name] = formatted

        metric, max_tokens = infer_metric(task_name, formatted)
        task_metrics[task_name] = {
            "metric": metric, "max_response_tokens": max_tokens}

    return task_data, task_definitions, task_families, task_metrics


@torch.no_grad()
def evaluate_tasks(model, tokenizer, eval_data, task_metrics,
                   device, max_examples=200, label="Eval"):
    """Evaluate on a set of SNI tasks and return per-task + macro scores."""
    print(f"\n=== {label}: {len(eval_data)} tasks ===")
    model.eval()
    all_scores = {}

    for tid, examples in eval_data.items():
        if tid not in task_metrics:
            continue

        eval_examples = examples[:max_examples]
        metric_name = task_metrics[tid]["metric"]
        max_tokens = task_metrics[tid]["max_response_tokens"]

        try:
            predictions = generate_predictions(
                model, tokenizer, eval_examples,
                max_new_tokens=max_tokens, device=device,
            )
            references = [ex["response"] for ex in eval_examples]
            metric_fn = METRIC_FNS.get(metric_name, accuracy)
            scores = metric_fn(predictions, references)
            all_scores[tid] = scores
            primary = list(scores.values())[0]
            print(f"  {tid}: {primary:.2f} ({metric_name})")
        except Exception as e:
            print(f"  {tid}: eval failed: {e}")
            continue

    if all_scores:
        primary_values = [list(v.values())[0] for v in all_scores.values()]
        macro = float(np.mean(primary_values))
        all_scores["macro_avg"] = macro
        print(f"\n  {label} macro: {macro:.2f} "
              f"(over {len(primary_values)} tasks)")

    return all_scores


def train_lora_sni41(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model + LoRA ──
    print(f"Loading backbone: {args.backbone}")
    model, tokenizer = load_backbone(args.backbone)
    print(f"Adding shared LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    model = add_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model = model.to(device)
    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ── Load SNI training data ──
    seen = set()
    train_task_list = [t for t in TRAIN_TASKS_SNI
                       if t not in seen and not seen.add(t)]

    train_data, train_defs, train_families, train_metrics = load_sni_data(
        train_task_list, split="train",
        max_per_task=args.max_per_task, cache_dir=args.sni_cache_dir,
    )
    task_ids = list(train_data.keys())
    print(f"\nTraining on {len(task_ids)} tasks, "
          f"{sum(len(v) for v in train_data.values()):,} examples")

    if not task_ids:
        print("ERROR: No valid training tasks. Exiting.")
        sys.exit(1)

    # ── Optimizer ──
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    max_steps = 2 if args.dry_run else args.max_steps
    warmup_steps = int(max_steps * args.warmup_ratio)
    warmup_sched = LinearLR(optimizer, start_factor=0.01,
                            total_iters=max(warmup_steps, 1))
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=max(max_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    # ── Training loop ──
    print(f"\nStarting shared LoRA training for {max_steps} steps...")
    model.train()
    global_step = 0
    accum_loss = 0.0
    t_start = time.time()

    dataloader = build_dataloader(
        train_data, args.microbatch_size,
        max_steps * args.gradient_accumulation_steps, args.seed)

    for step_idx, (task_id, examples) in enumerate(dataloader):
        batch = tokenize_batch(tokenizer, examples, args.max_seq_length)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()
        accum_loss += loss.item()

        if (step_idx + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 100 == 0 or global_step == 1:
                elapsed = time.time() - t_start
                fam = train_families.get(task_id, "?")
                print(f"  Step {global_step}/{max_steps} | "
                      f"Loss: {accum_loss:.4f} | "
                      f"Task: {task_id} ({fam}) | "
                      f"Time: {elapsed:.0f}s")
                accum_loss = 0.0

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({global_step} steps)")

    # ── Save ──
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    print(f"Saved model to {final_path}")

    # ==================================================================
    # Evaluation Phase
    # ==================================================================
    print("\n" + "=" * 70)
    print("EVALUATION PHASE")
    print("=" * 70)

    # ── 1. Evaluate on trained tasks (validation split) ──
    val_data, _, _, val_metrics = load_sni_data(
        task_ids, split="validation",
        max_per_task=args.max_per_task, cache_dir=args.sni_cache_dir,
    )
    for tid in val_data:
        if tid not in val_metrics:
            val_metrics[tid] = train_metrics.get(
                tid, {"metric": "accuracy", "max_response_tokens": 16})

    trained_scores = evaluate_tasks(
        model, tokenizer, val_data, val_metrics, device,
        max_examples=args.max_eval_examples,
        label="Trained Tasks (validation)",
    )

    # ── 2. Cold-start on held-out tasks (shared LoRA applied unchanged) ──
    holdout_list = [t for t in HOLDOUT_TASKS_SNI if t not in set(task_ids)]
    print(f"\nHeld-out tasks: {len(holdout_list)}")

    holdout_data, holdout_defs, holdout_families, holdout_metrics = load_sni_data(
        holdout_list, split="validation",
        max_per_task=args.max_per_task, cache_dir=args.sni_cache_dir,
    )

    coldstart_scores = evaluate_tasks(
        model, tokenizer, holdout_data, holdout_metrics, device,
        max_examples=args.max_eval_examples,
        label="Cold-Start (shared LoRA, no task-specific training)",
    )

    # ── 3. Also evaluate frozen baseline on held-out tasks ──
    print("\n--- Frozen baseline on held-out tasks ---")
    frozen_model, frozen_tok = load_backbone(args.backbone)
    frozen_model = frozen_model.to(device)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False

    frozen_scores = evaluate_tasks(
        frozen_model, frozen_tok, holdout_data, holdout_metrics, device,
        max_examples=args.max_eval_examples,
        label="Frozen Baseline (held-out)",
    )
    del frozen_model
    torch.cuda.empty_cache()

    # ==================================================================
    # Results summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if coldstart_scores and frozen_scores:
        print(f"\n  {'Task':<25s} {'Frozen':>8s} {'LoRA':>8s} {'Delta':>8s}")
        for tid in sorted(holdout_data.keys()):
            f_val = list(frozen_scores[tid].values())[0] if tid in frozen_scores else float('nan')
            l_val = list(coldstart_scores[tid].values())[0] if tid in coldstart_scores else float('nan')
            delta = l_val - f_val
            print(f"  {tid:<25s} {f_val:>8.1f} {l_val:>8.1f} {delta:>+8.1f}")
        f_macro = frozen_scores.get("macro_avg", 0)
        l_macro = coldstart_scores.get("macro_avg", 0)
        print(f"  {'MACRO':<25s} {f_macro:>8.1f} {l_macro:>8.1f} {l_macro - f_macro:>+8.1f}")

    results = {
        "experiment": "shared_lora_sni41_coldstart",
        "backbone": args.backbone,
        "lora_rank": args.lora_rank,
        "trainable_params": trainable,
        "num_train_tasks": len(task_ids),
        "num_holdout_tasks": len(holdout_data),
        "max_steps": max_steps,
        "trained_scores": trained_scores,
        "coldstart_scores": coldstart_scores,
        "frozen_scores": frozen_scores,
        "training_time_seconds": total_time,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {results_path}")

    print("\n=== RESULTS JSON ===")
    print(json.dumps(results, indent=2, default=str))
    print("=== END RESULTS ===")

    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    train_lora_sni41(args)
