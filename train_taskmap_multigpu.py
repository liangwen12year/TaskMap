"""
Multi-GPU TaskMap training using Accelerate.

Distributes microbatches across GPUs while keeping TaskMap components
(task codes, mapper, router) synchronized. The frozen backbone is
replicated automatically by Accelerate's DDP wrapper.

Usage:
  # Single GPU (same as train_taskmap_scaled.py)
  python train_taskmap_multigpu.py --backbone Qwen/Qwen2.5-1.5B --max_steps 12000

  # Multi-GPU via accelerate
  accelerate launch --num_processes 6 train_taskmap_multigpu.py \
    --backbone Qwen/Qwen2.5-1.5B --max_steps 12000

  # Or with torchrun
  torchrun --nproc_per_node 6 train_taskmap_multigpu.py \
    --backbone Qwen/Qwen2.5-1.5B --max_steps 12000
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone, count_parameters
from models.taskmap_model import TaskMapModel, TaskMapConfig
from models.ffn_hooks import TaskMapHookManager
from data.task_collection import (
    TRAIN_TASKS_SNI, HOLDOUT_TASKS_SNI,
    load_sni_dataset, filter_sni_tasks, format_sni_examples, get_task_family,
)
from data.sampler import build_dataloader
from losses import TaskMapLossComputer
from train import tokenize_batch, set_seed

try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU TaskMap Training")
    parser.add_argument("--config", type=str, default="configs/taskmap_reference.yaml")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_steps", type=int, default=12000)
    parser.add_argument("--active_fraction", type=float, default=0.5)
    parser.add_argument("--code_dim", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_per_task", type=int, default=2000)
    parser.add_argument("--max_eval_examples", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default="outputs/taskmap_multigpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze_mapper", action="store_true")
    parser.add_argument("--mapping_loss", action="store_true")
    parser.add_argument("--shared_projector", action="store_true")
    parser.add_argument("--global_code", action="store_true")
    parser.add_argument("--sni_cache_dir", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_sni", action="store_true", default=True,
                        help="Use SNI tasks (default). Set --no_sni for original 10 tasks")
    parser.add_argument("--no_sni", dest="use_sni", action="store_false")
    return parser.parse_args()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def print_main(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def load_sni_data_for_training(task_list, max_per_task, cache_dir=None):
    """Load SNI tasks for training. Returns task_data, definitions, families, metrics."""
    full_ds = load_sni_dataset(cache_dir)

    # Validate task names
    available_tasks = set()
    for ex in full_ds:
        available_tasks.add(ex['task_name'])
    valid_tasks = [t for t in task_list if t in available_tasks]
    missing = [t for t in task_list if t not in available_tasks]
    if missing:
        print_main(f"  WARNING: {len(missing)} tasks not found: {missing[:5]}...")
    print_main(f"  Found {len(valid_tasks)}/{len(task_list)} tasks")

    # Filter and format
    raw_data = filter_sni_tasks(full_ds, valid_tasks, max_per_task)

    task_data = {}
    definitions = {}
    families = {}
    metrics = {}

    for tid in valid_tasks:
        if tid not in raw_data:
            continue
        examples = format_sni_examples(tid, raw_data[tid], split="train")
        if not examples:
            continue

        task_data[tid] = examples
        definitions[tid] = raw_data[tid][0].get('definition', '')[:200]

        # Infer family
        defn = definitions[tid].lower()
        if any(w in defn for w in ['classif', 'detect', 'label', 'categor', 'spam']):
            fam = 'classification'
        elif any(w in defn for w in ['answer', 'question', 'comprehension']):
            fam = 'question_answering'
        elif any(w in defn for w in ['summar', 'title']):
            fam = 'summarization'
        elif any(w in defn for w in ['translat', 'paraphras', 'simplif']):
            fam = 'paraphrase'
        elif any(w in defn for w in ['entail', 'inference', 'nli']):
            fam = 'nli'
        elif any(w in defn for w in ['math', 'arithmetic', 'calcul', 'number']):
            fam = 'reasoning'
        elif any(w in defn for w in ['generat', 'story', 'write', 'creat']):
            fam = 'generation'
        else:
            fam = 'other'
        families[tid] = fam

        # Infer metric
        avg_resp_len = np.mean([len(ex['response'].split()) for ex in examples[:50]])
        if avg_resp_len < 5:
            metrics[tid] = {"metric": "accuracy", "max_response_tokens": 32}
        else:
            metrics[tid] = {"metric": "rouge_l", "max_response_tokens": 128}

        print_main(f"  {tid}: {len(examples)} examples, family={fam}, metric={metrics[tid]['metric']}")

    return task_data, definitions, families, metrics


def train_multigpu(args):
    set_seed(args.seed)

    # Initialize accelerator
    if HAS_ACCELERATE:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="bf16",
        )
        device = accelerator.device
        is_main = accelerator.is_main_process
    else:
        accelerator = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_main = True

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load backbone
    print_main(f"Loading frozen backbone: {args.backbone}")
    backbone, tokenizer = load_backbone(args.backbone)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Load data (only on main process, then broadcast)
    print_main("\nLoading training data...")
    if args.use_sni:
        seen = set()
        task_list = [t for t in TRAIN_TASKS_SNI if t not in seen and not seen.add(t)]
        train_data, definitions, families, metrics = load_sni_data_for_training(
            task_list, args.max_per_task, args.sni_cache_dir
        )
    else:
        from data.config import KNOWN_TASKS
        from data.download import download_task
        from data.format import format_all_tasks
        datasets = {}
        for tid, meta in KNOWN_TASKS.items():
            ds = download_task(tid, meta)
            if ds is not None:
                datasets[tid] = ds
        train_data = format_all_tasks(datasets, split="train")
        definitions = {tid: KNOWN_TASKS[tid]["descriptions"][0] for tid in train_data}
        families = {tid: KNOWN_TASKS[tid]["family"] for tid in train_data}
        metrics = {tid: {"metric": KNOWN_TASKS[tid]["metric"],
                         "max_response_tokens": KNOWN_TASKS[tid]["max_response_tokens"]}
                   for tid in train_data}

    task_ids = list(train_data.keys())
    print_main(f"Training tasks: {len(task_ids)}, Total examples: {sum(len(v) for v in train_data.values()):,}")

    if not task_ids:
        print_main("ERROR: No valid tasks. Exiting.")
        return

    # Setup TaskMap
    tm_config = TaskMapConfig.from_backbone(
        args.backbone,
        block_size=args.block_size,
        active_fraction=args.active_fraction,
        code_dim=args.code_dim,
        rank=args.rank,
        mapper_hidden=512,
        total_steps=args.max_steps,
    )

    taskmap = TaskMapModel(tm_config, num_tasks=len(task_ids),
                           freeze_mapper=not args.unfreeze_mapper,
                           shared_projector=args.shared_projector,
                           global_code=args.global_code).to(device)
    taskmap.register_tasks(task_ids)

    # Cache description embeddings
    print_main("Computing description embeddings...")
    for tid in task_ids:
        desc = definitions.get(tid, tid)
        embed = taskmap.task_code.compute_description_embedding(
            backbone, tokenizer, desc, device
        )
        taskmap.cache_description(tid, embed)

    hook_manager = TaskMapHookManager(backbone, taskmap, block_size=args.block_size)

    # Optimizer
    param_groups = []
    code_params = [p for n, p in taskmap.task_code.named_parameters()
                   if p.requires_grad and "projector" not in n]
    proj_params = [p for n, p in taskmap.task_code.named_parameters()
                   if p.requires_grad and "projector" in n]
    param_groups.append({"params": code_params, "lr": 2e-3})
    param_groups.append({"params": proj_params, "lr": 2e-4})
    if args.unfreeze_mapper:
        mapper_params = [p for p in taskmap.mapper_bank.parameters() if p.requires_grad]
        param_groups.append({"params": mapper_params, "lr": 2e-4})

    optimizer = AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.95))

    max_steps = 2 if args.dry_run else args.max_steps
    warmup_steps = int(max_steps * 0.03)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(max_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    # Build family pairs
    family_pairs = []
    for i, t1 in enumerate(task_ids):
        for t2 in task_ids[i+1:]:
            if families.get(t1) == families.get(t2):
                family_pairs.append((t1, t2))

    loss_computer = TaskMapLossComputer(
        tm_config, family_pairs, families,
        lambda_bud=0.05, lambda_topo=0.0, lambda_bal=0.01,
        lambda_stab=1e-3, lambda_sm=1e-3, lambda_align=1e-4,
        active_mapping_loss=args.mapping_loss,
    )

    # Wrap with accelerator
    if accelerator:
        taskmap, optimizer, scheduler = accelerator.prepare(
            taskmap, optimizer, scheduler
        )

    # Training loop
    grad_accum = args.gradient_accumulation_steps
    max_seq = args.max_seq_length
    microbatch_size = args.microbatch_size

    print_main(f"\nStarting training for {max_steps} steps on {torch.cuda.device_count()} GPUs...")
    print_main(f"  Effective batch = {microbatch_size} × {grad_accum} × {torch.cuda.device_count()} GPUs")

    global_step = 0
    accum_loss = 0.0
    all_route_masks = {}
    t_start = time.time()

    dataloader = build_dataloader(train_data, microbatch_size,
                                  max_steps * grad_accum, args.seed)

    for step_idx, (task_id, examples) in enumerate(dataloader):
        # Route computation (same on all ranks since TaskMap is synced)
        tm = taskmap.module if hasattr(taskmap, 'module') else taskmap
        tm.clear_route_cache()
        routes = tm.compute_route(task_id, device)
        masks_for_task = [r['mask'].detach() for r in routes]
        all_route_masks[task_id] = masks_for_task
        hook_manager.activate_for_task(task_id, device)

        batch = tokenize_batch(tokenizer, examples, max_seq)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = backbone(**batch)
        task_loss = outputs.loss

        is_warmup = tm.router.is_warmup()
        total_loss, loss_dict = loss_computer.compute(
            task_loss, tm, task_id, all_route_masks, is_warmup
        )

        scaled_loss = total_loss / grad_accum
        if accelerator:
            accelerator.backward(scaled_loss)
        else:
            scaled_loss.backward()
        accum_loss += scaled_loss.item()

        if (step_idx + 1) % grad_accum == 0:
            if accelerator:
                accelerator.clip_grad_norm_(tm.trainable_parameters(), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(tm.trainable_parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tm.step()
            global_step += 1

            if (global_step % 100 == 0 or global_step == 1) and is_main:
                elapsed = time.time() - t_start
                temp = tm.router.get_temperature()
                phase = "warmup" if is_warmup else f"temp={temp:.3f}"
                fam = families.get(task_id, "?")
                print(f"  Step {global_step}/{max_steps} | "
                      f"Loss: {accum_loss:.4f} | Task: {task_id} ({fam}) | "
                      f"Phase: {phase} | Time: {elapsed:.0f}s")
                accum_loss = 0.0

            if not args.dry_run and global_step % args.save_every == 0 and is_main:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    "task_code_state": tm.task_code.state_dict(),
                    "mapper_state": tm.mapper_bank.state_dict(),
                    "step": global_step,
                    "task_ids": task_ids,
                    "families": families,
                }, os.path.join(save_path, "taskmap_state.pt"))

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print_main(f"\nTraining complete in {total_time:.1f}s ({global_step} steps)")

    # Save final
    if is_main:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        torch.save({
            "task_code_state": tm.task_code.state_dict(),
            "mapper_state": tm.mapper_bank.state_dict(),
            "step": global_step,
            "task_ids": task_ids,
            "families": families,
            "metrics": metrics,
        }, os.path.join(final_path, "taskmap_state.pt"))
        print(f"Saved final model to {final_path}")

    # Evaluation (main process only)
    if is_main:
        from eval import evaluate_task
        print("\n=== Evaluating trained tasks ===")
        backbone.eval()
        all_scores = {}
        max_eval = args.max_eval_examples

        for tid in task_ids:
            if tid not in train_data:
                continue
            # Use validation split
            val_examples = format_sni_examples(tid, filter_sni_tasks(
                load_sni_dataset(args.sni_cache_dir), [tid], args.max_per_task
            ).get(tid, []), split="validation")
            if not val_examples:
                continue
            if len(val_examples) > max_eval:
                val_examples = val_examples[:max_eval]

            hook_manager.activate_for_task(tid, device)
            metric_name = metrics[tid]["metric"]
            max_tokens = metrics[tid]["max_response_tokens"]
            scores = evaluate_task(backbone, tokenizer, tid, val_examples,
                                   metric_name, max_tokens, device)
            all_scores[tid] = scores
            print(f"  {tid}: {scores}")

        primary = [list(v.values())[0] for v in all_scores.values() if isinstance(v, dict)]
        macro = np.mean(primary) if primary else 0.0
        print(f"\nTrained tasks macro: {macro:.2f}")

        # Cold-start evaluation
        print("\n=== Cold-Start Evaluation ===")
        holdout_list = [t for t in HOLDOUT_TASKS_SNI if t not in set(task_ids)]
        full_ds = load_sni_dataset(args.sni_cache_dir)
        holdout_raw = filter_sni_tasks(full_ds, holdout_list, args.max_per_task)

        cold_scores = {}
        for tid in holdout_list:
            if tid not in holdout_raw:
                continue
            examples = format_sni_examples(tid, holdout_raw[tid], split="validation")
            if not examples:
                continue
            if len(examples) > max_eval:
                examples = examples[:max_eval]

            desc = holdout_raw[tid][0].get('definition', tid)[:200]
            embed = tm.task_code.compute_description_embedding(
                backbone, tokenizer, desc, device
            )
            tm.cache_description(tid, embed)
            tm.clear_route_cache()
            try:
                tm.compute_route(tid, device)
                hook_manager.activate_for_task(tid, device)
            except Exception as e:
                print(f"  Skipping {tid}: {e}")
                continue

            avg_len = np.mean([len(ex['response'].split()) for ex in examples[:20]])
            metric_name = "accuracy" if avg_len < 5 else "rouge_l"
            max_tokens = 32 if avg_len < 5 else 128

            scores = evaluate_task(backbone, tokenizer, tid, examples,
                                   metric_name, max_tokens, device)
            cold_scores[tid] = scores
            print(f"  {tid}: {scores}")

        cold_primary = [list(v.values())[0] for v in cold_scores.values() if isinstance(v, dict)]
        cold_macro = np.mean(cold_primary) if cold_primary else 0.0
        print(f"\nCold-start macro: {cold_macro:.2f}")

        results = {
            "experiment": "taskmap_multigpu",
            "backbone": args.backbone,
            "num_gpus": torch.cuda.device_count(),
            "num_train_tasks": len(task_ids),
            "trained_scores": all_scores,
            "trained_macro": float(macro),
            "cold_start_scores": cold_scores,
            "cold_start_macro": float(cold_macro),
            "training_time_seconds": total_time,
        }
        print("\n=== RESULTS JSON ===")
        print(json.dumps(results, indent=2, default=str))
        print("=== END RESULTS ===")

        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    hook_manager.remove_all()


if __name__ == "__main__":
    args = parse_args()
    train_multigpu(args)
