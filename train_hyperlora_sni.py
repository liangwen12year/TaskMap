"""
HyperLoRA baseline on SNI tasks with cold-start evaluation.

Adapts train_hyperlora.py to use the 41 SNI training tasks and evaluate
cold-start generalization on 20 held-out tasks.  For held-out tasks, the
HyperLoRA generator produces LoRA factors directly from the task's
description embedding -- no per-task training is needed.

Usage:
  python train_hyperlora_sni.py --backbone Qwen/Qwen2.5-1.5B --max_steps 12000
  python train_hyperlora_sni.py --dry_run  # 2 steps on CPU
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone, count_parameters
from models.task_code import TaskCodeModule
from data.task_collection import (
    TRAIN_TASKS_SNI,
    HOLDOUT_TASKS_SNI,
    load_sni_dataset,
    filter_sni_tasks,
    format_sni_examples,
    get_task_family,
)
from data.sampler import build_dataloader
from train import tokenize_batch, set_seed
from eval import METRIC_FNS, generate_predictions, accuracy, rouge_l
from train_hyperlora import HyperLoRAGenerator, HyperLoRAHook


# ---------------------------------------------------------------------------
# Helpers (same as train_taskmap_scaled.py)
# ---------------------------------------------------------------------------

def validate_sni_tasks(full_dataset, requested_tasks):
    """Check which of the requested task names actually exist in the dataset."""
    available = set()
    for ex in full_dataset:
        available.add(ex["task_name"])
    valid = [t for t in requested_tasks if t in available]
    missing = [t for t in requested_tasks if t not in available]
    return valid, missing


def infer_metric(task_name, examples, threshold_words=5):
    """Determine evaluation metric for an SNI task automatically."""
    if not examples:
        return "accuracy", 16
    total_words = 0
    for ex in examples[:200]:
        total_words += len(ex["response"].split())
    avg_words = total_words / min(len(examples), 200)
    if avg_words < threshold_words:
        return "accuracy", 16
    else:
        return "rouge_l", 128


def load_sni_data(task_list, split, max_per_task, cache_dir=None):
    """Load SNI tasks, validate, format, and return task data dict."""
    print(f"\nLoading SNI dataset ({len(task_list)} requested tasks, split={split})...")
    full_ds = load_sni_dataset(cache_dir)

    valid_tasks, missing_tasks = validate_sni_tasks(full_ds, task_list)
    if missing_tasks:
        print(f"  WARNING: {len(missing_tasks)} tasks not found in dataset:")
        for t in missing_tasks:
            print(f"    - {t}")
    print(f"  Valid tasks: {len(valid_tasks)}/{len(task_list)}")

    raw_data = filter_sni_tasks(full_ds, valid_tasks, max_per_task)

    task_data = {}
    task_definitions = {}
    task_metrics = {}

    for task_name in valid_tasks:
        if task_name not in raw_data or not raw_data[task_name]:
            print(f"  {task_name}: no examples after filtering, skipping")
            continue

        definition = raw_data[task_name][0].get("definition", "")
        task_definitions[task_name] = definition

        formatted = format_sni_examples(task_name, raw_data[task_name], split)
        if not formatted:
            print(f"  {task_name}: no examples after formatting, skipping")
            continue

        task_data[task_name] = formatted

        metric, max_tokens = infer_metric(task_name, formatted)
        task_metrics[task_name] = {"metric": metric, "max_response_tokens": max_tokens}

        print(f"  {task_name}: {len(formatted)} examples, metric={metric}")

    print(f"\nLoaded {len(task_data)} tasks with "
          f"{sum(len(v) for v in task_data.values()):,} total examples")

    return task_data, task_definitions, task_metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hyperlora_tasks(backbone_model, tokenizer, generator, hooks,
                             task_embeds, eval_data, task_metrics,
                             num_layers, device, max_examples=200,
                             label="Eval"):
    """Evaluate a set of tasks using HyperLoRA-generated factors."""
    print(f"\n=== {label}: {len(eval_data)} tasks ===")
    backbone_model.eval()
    generator.eval()
    all_scores = {}

    for tid, examples in eval_data.items():
        if tid not in task_metrics or tid not in task_embeds:
            continue

        eval_examples = examples[:max_examples]
        metric_name = task_metrics[tid]["metric"]
        max_tokens = task_metrics[tid]["max_response_tokens"]

        # Activate HyperLoRA factors for this task
        embed = task_embeds[tid]
        for l in range(num_layers):
            factors = generator(embed, l)
            hooks[l].set_factors(factors)

        try:
            predictions = generate_predictions(
                backbone_model, tokenizer, eval_examples,
                max_new_tokens=max_tokens, device=device,
            )
            references = [ex["response"] for ex in eval_examples]

            metric_fn = METRIC_FNS.get(metric_name)
            if metric_fn is None:
                metric_fn = accuracy if metric_name == "accuracy" else rouge_l

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
        print(f"\n  {label} macro average: {macro:.2f} "
              f"(over {len(primary_values)} tasks)")
    else:
        print(f"  No tasks evaluated successfully for {label}")

    # Clear factors after evaluation
    for hook in hooks:
        hook.set_factors(None)

    return all_scores


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="HyperLoRA SNI Training + Cold-Start Eval")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_steps", type=int, default=12000)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--code_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_per_task", type=int, default=2000,
                        help="Max examples per task from SNI")
    parser.add_argument("--max_eval_examples", type=int, default=200,
                        help="Max evaluation examples per task")
    parser.add_argument("--sni_cache_dir", type=str, default=None,
                        help="Cache directory for SNI dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/hyperlora_sni")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--eval_trecdl", action="store_true",
                        help="Run TREC-DL 2019 reranking after training")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_hyperlora_sni(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load backbone (frozen) ──
    print(f"Loading backbone: {args.backbone}")
    backbone, tokenizer = load_backbone(args.backbone)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    config = backbone.config
    num_layers = config.num_hidden_layers
    model_dim = config.hidden_size
    ffn_dim = config.intermediate_size
    embed_dim = model_dim

    print(f"Architecture: {num_layers} layers, d={model_dim}, d_ff={ffn_dim}")

    # ── Build HyperLoRA generator ──
    generator = HyperLoRAGenerator(
        embed_dim=embed_dim, num_layers=num_layers, rank=args.rank,
        model_dim=model_dim, ffn_dim=ffn_dim,
        hidden_dim=args.hidden_dim, code_dim=args.code_dim,
    ).to(device)

    trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"HyperLoRA generator: {trainable:,} trainable parameters")

    # ── Load SNI training data ──
    seen = set()
    train_task_list = []
    for t in TRAIN_TASKS_SNI:
        if t not in seen:
            seen.add(t)
            train_task_list.append(t)

    train_data, train_definitions, train_metrics = load_sni_data(
        train_task_list, split="train",
        max_per_task=args.max_per_task, cache_dir=args.sni_cache_dir,
    )
    task_ids = list(train_data.keys())
    total_examples = sum(len(v) for v in train_data.values())
    print(f"\nTraining tasks: {len(task_ids)}")
    print(f"Total training examples: {total_examples:,}")

    if len(task_ids) == 0:
        print("ERROR: No valid training tasks. Exiting.")
        sys.exit(1)

    # ── Compute description embeddings for training tasks ──
    task_code = TaskCodeModule(
        num_layers=1, embed_dim=embed_dim, code_dim=embed_dim, num_tasks=0
    )

    print(f"Computing description embeddings for {len(task_ids)} tasks...")
    task_embeds = {}
    for tid in task_ids:
        definition = train_definitions.get(tid, f"Complete the following task: {tid}")
        desc = definition[:200] if definition else f"Complete the following task: {tid}"
        embed = task_code.compute_description_embedding(
            backbone, tokenizer, desc, device
        )
        task_embeds[tid] = embed
        print(f"  {tid}: norm={embed.norm():.3f}")

    # ── Install hooks ──
    if hasattr(backbone, 'model'):
        mlp_layers = [backbone.model.layers[i].mlp for i in range(num_layers)]
    else:
        mlp_layers = [backbone.transformer.h[i].mlp for i in range(num_layers)]

    hooks = []
    for l in range(num_layers):
        hook = HyperLoRAHook(mlp_layers[l], l, model_dim, ffn_dim)
        hooks.append(hook)

    # ── Optimizer ──
    optimizer = AdamW(generator.parameters(), lr=args.lr,
                      weight_decay=0.01, betas=(0.9, 0.95))
    max_steps = 2 if args.dry_run else args.max_steps
    warmup_steps = int(max_steps * 0.03)
    warmup_sched = LinearLR(optimizer, start_factor=0.01,
                            total_iters=max(warmup_steps, 1))
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=max(max_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    # ── Training loop ──
    print(f"\nStarting HyperLoRA SNI training for {max_steps} steps...")
    print(f"  Tasks: {len(task_ids)}")
    print(f"  Rank: {args.rank}")
    generator.train()
    global_step = 0
    accum_loss = 0.0
    t_start = time.time()

    dataloader = build_dataloader(train_data, args.microbatch_size,
                                  max_steps * args.gradient_accumulation_steps,
                                  args.seed)

    for step_idx, (task_id, examples) in enumerate(dataloader):
        embed = task_embeds[task_id]
        for l in range(num_layers):
            factors = generator(embed, l)
            hooks[l].set_factors(factors)

        batch = tokenize_batch(tokenizer, examples, args.max_seq_length)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = backbone(**batch)
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()
        accum_loss += loss.item()

        if (step_idx + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 100 == 0 or global_step == 1:
                elapsed = time.time() - t_start
                family = get_task_family(task_id)
                print(f"  Step {global_step}/{max_steps} | "
                      f"Loss: {accum_loss:.4f} | "
                      f"Task: {task_id} ({family}) | Time: {elapsed:.0f}s")
                accum_loss = 0.0

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({global_step} steps)")

    # ==================================================================
    # Evaluation Phase
    # ==================================================================
    print("\n" + "=" * 70)
    print("EVALUATION PHASE")
    print("=" * 70)

    # ── 1. Evaluate on trained tasks (validation split) ──
    val_data, _, val_metrics = load_sni_data(
        task_ids, split="validation",
        max_per_task=args.max_per_task, cache_dir=args.sni_cache_dir,
    )
    for tid in val_data:
        if tid not in val_metrics:
            val_metrics[tid] = train_metrics.get(tid, {"metric": "accuracy",
                                                       "max_response_tokens": 16})

    trained_scores = evaluate_hyperlora_tasks(
        backbone, tokenizer, generator, hooks,
        task_embeds, val_data, val_metrics,
        num_layers, device,
        max_examples=args.max_eval_examples,
        label="Trained Tasks (validation)",
    )

    # ── 2. Cold-start evaluation on held-out tasks ──
    holdout_list = [t for t in HOLDOUT_TASKS_SNI if t not in set(task_ids)]
    print(f"\nHeld-out task list: {len(holdout_list)} tasks")

    holdout_data, holdout_defs, holdout_metrics = load_sni_data(
        holdout_list, split="validation",
        max_per_task=args.max_per_task, cache_dir=args.sni_cache_dir,
    )

    # Compute description embeddings for held-out tasks (cold-start)
    print(f"\nComputing cold-start description embeddings for "
          f"{len(holdout_data)} held-out tasks...")
    holdout_embeds = {}
    for tid in holdout_data:
        definition = holdout_defs.get(tid, f"Complete the following task: {tid}")
        desc = definition[:200] if definition else f"Complete the following task: {tid}"
        embed = task_code.compute_description_embedding(
            backbone, tokenizer, desc, device
        )
        holdout_embeds[tid] = embed
        print(f"  {tid}: norm={embed.norm():.3f}")

    coldstart_scores = evaluate_hyperlora_tasks(
        backbone, tokenizer, generator, hooks,
        holdout_embeds, holdout_data, holdout_metrics,
        num_layers, device,
        max_examples=args.max_eval_examples,
        label="Cold-Start (held-out tasks)",
    )

    # ==================================================================
    # Results Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if trained_scores:
        print(f"\n  Trained tasks macro: "
              f"{trained_scores.get('macro_avg', 0):.2f}")
    if coldstart_scores:
        print(f"  Cold-start macro:   "
              f"{coldstart_scores.get('macro_avg', 0):.2f}")

    # Per-family breakdown for trained tasks
    if trained_scores:
        family_scores = defaultdict(list)
        for tid, scores in trained_scores.items():
            if tid == "macro_avg":
                continue
            fam = get_task_family(tid)
            family_scores[fam].append(list(scores.values())[0])

        print("\nTrained tasks - per family:")
        for fam in sorted(family_scores.keys()):
            vals = family_scores[fam]
            print(f"  {fam:20s}: {np.mean(vals):.2f} (n={len(vals)})")

    # Per-family breakdown for cold-start
    if coldstart_scores:
        cs_family_scores = defaultdict(list)
        for tid, scores in coldstart_scores.items():
            if tid == "macro_avg":
                continue
            fam = get_task_family(tid)
            cs_family_scores[fam].append(list(scores.values())[0])

        print("\nCold-start tasks - per family:")
        for fam in sorted(cs_family_scores.keys()):
            vals = cs_family_scores[fam]
            print(f"  {fam:20s}: {np.mean(vals):.2f} (n={len(vals)})")

    # Clean up hooks
    for hook in hooks:
        hook.set_factors(None)

    # Build full results dict
    results = {
        "experiment": "hyperlora_sni",
        "backbone": args.backbone,
        "rank": args.rank,
        "trainable_params": trainable,
        "num_train_tasks": len(task_ids),
        "num_holdout_tasks": len(holdout_data),
        "max_steps": max_steps,
        "seed": args.seed,
        "trained_scores": trained_scores,
        "cold_start_scores": coldstart_scores,
        "task_ids": task_ids,
        "training_time_seconds": total_time,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {results_path}")

    print("\n=== RESULTS JSON ===")
    print(json.dumps(results, indent=2, default=str))
    print("=== END RESULTS ===")

    # ── TREC-DL 2019 reranking (optional) ──
    if args.eval_trecdl:
        try:
            from eval_trecdl import (
                load_candidates, load_qrels, score_pairs,
                evaluate_rankings, eval_bm25_baseline, TASK_DESC,
            )
            print("\n=== TREC-DL 2019 Reranking (HyperLoRA cold-start) ===")
            candidates = load_candidates()
            qrels = load_qrels()

            # Activate HyperLoRA factors for the search task description
            search_embed = task_code.compute_description_embedding(
                backbone, tokenizer, TASK_DESC[:200], device
            )
            for l in range(num_layers):
                factors = generator(search_embed, l)
                hooks[l].set_factors(factors)

            # Score and evaluate
            bm25_res = eval_bm25_baseline(candidates, qrels)
            print(f"  BM25: nDCG@10={bm25_res['ndcg@10']:.4f}")
            rankings = score_pairs(backbone, tokenizer, candidates, device)
            trecdl_res = evaluate_rankings(rankings, qrels)
            print(f"  HyperLoRA: nDCG@10={trecdl_res['ndcg@10']:.4f}  "
                  f"MRR@10={trecdl_res['mrr@10']:.4f}  "
                  f"MAP@100={trecdl_res['map@100']:.4f}")

            # Clean up
            for hook in hooks:
                hook.set_factors(None)

            trecdl_out = {
                "method": "hyperlora", "seed": args.seed,
                "bm25": bm25_res, "hyperlora": trecdl_res,
            }
            trecdl_path = os.path.join("outputs/trecdl",
                                       f"trecdl_hyperlora_s{args.seed}.json")
            os.makedirs("outputs/trecdl", exist_ok=True)
            with open(trecdl_path, "w") as f:
                json.dump(trecdl_out, f, indent=2)
            print(f"  Saved to {trecdl_path}")
        except Exception as e:
            print(f"  TREC-DL eval failed: {e}")
            import traceback; traceback.print_exc()

    print("=== DONE ===")


if __name__ == "__main__":
    args = parse_args()
    train_hyperlora_sni(args)
