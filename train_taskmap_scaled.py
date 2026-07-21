"""
Scaled TaskMap training with 50+ SNI tasks and 20 held-out tasks.

Extends train_taskmap.py to use Super-NaturalInstructions (SNI) as the task
source, replacing the 12 KNOWN_TASKS with ~50 diverse training tasks and
evaluating cold-start generalization on 20 held-out tasks.

Key differences from train_taskmap.py:
1. Loads tasks from SNI via data/task_collection.py instead of data/config.py
2. Infers metric per task automatically (accuracy for short outputs, ROUGE-L otherwise)
3. Builds family pairs and family maps from SNI metadata
4. Validates that requested task names actually exist in the dataset
5. Splits SNI data 80/20 since SNI has no predefined train/validation splits
6. Evaluates on both trained tasks and held-out tasks for cold-start analysis

Usage:
  python train_taskmap_scaled.py --backbone Qwen/Qwen2.5-1.5B --max_steps 12000
  python train_taskmap_scaled.py --dry_run  # 2 steps on CPU
"""

import os
import sys
import time
import json
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone, count_parameters
from models.taskmap_model import TaskMapModel, TaskMapConfig
from models.ffn_hooks import TaskMapHookManager
from data.task_collection import (
    TRAIN_TASKS_SNI,
    HOLDOUT_TASKS_SNI,
    SNI_FAMILY_MAP,
    load_sni_dataset,
    filter_sni_tasks,
    format_sni_examples,
    get_task_family,
)
from data.sampler import build_dataloader
from losses import TaskMapLossComputer
from train import tokenize_batch, set_seed
from eval import METRIC_FNS, generate_predictions, accuracy, rouge_l


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_sni_tasks(full_dataset, requested_tasks):
    """Check which of the requested task names actually exist in the dataset.

    Returns (valid_tasks, missing_tasks) where valid_tasks preserves the
    original order.
    """
    available = set()
    for ex in full_dataset:
        available.add(ex["task_name"])
    valid = [t for t in requested_tasks if t in available]
    missing = [t for t in requested_tasks if t not in available]
    return valid, missing


def infer_metric(task_name, examples, threshold_words=5):
    """Determine the evaluation metric for an SNI task automatically.

    Heuristic: if the average response length is < threshold_words we treat
    it as classification (accuracy).  Otherwise ROUGE-L for generation.
    """
    if not examples:
        return "accuracy", 16

    total_words = 0
    for ex in examples[:200]:  # sample a subset for speed
        total_words += len(ex["response"].split())
    avg_words = total_words / min(len(examples), 200)

    if avg_words < threshold_words:
        return "accuracy", 16
    else:
        return "rouge_l", 128


def build_family_pairs_from_map(task_ids, task_families):
    """Build a list of (t1, t2) pairs within the same family."""
    family_to_tasks = defaultdict(list)
    for tid in task_ids:
        fam = task_families.get(tid, "other")
        family_to_tasks[fam].append(tid)

    pairs = []
    for fam, tasks in family_to_tasks.items():
        for i, t1 in enumerate(tasks):
            for t2 in tasks[i + 1:]:
                pairs.append((t1, t2))
    return pairs


def infer_family_from_definition(task_name, definition):
    """Try to assign a family based on the task definition text.

    Falls back to get_task_family (name-based) if definition matching fails.
    """
    defn_lower = definition.lower() if definition else ""

    # Keyword-based heuristics ordered from most to least specific
    keyword_map = [
        (["classify", "classification", "positive or negative",
          "sentiment", "toxic", "hate", "spam", "label", "yes or no",
          "true or false", "entailment"], "classification"),
        (["translate", "translation"], "translation"),
        (["summar", "title for"], "summarization"),
        (["answer the question", "reading comprehension", "question answering",
          "based on the passage", "based on the context"], "question_answering"),
        (["paraphrase", "rephrase", "rewrite"], "paraphrase"),
        (["extract", "named entity", "relation", "keyword", "tag"], "extraction"),
        (["generate", "write a", "compose", "create a", "story", "dialogue"], "generation"),
        (["cause", "effect", "reason", "logic", "common sense",
          "analogy", "math", "comput", "numerical"], "reasoning"),
        (["entail", "contradict", "inference"], "nli"),
        (["coreference", "word meaning", "linguistic", "part of speech"], "linguistics"),
        (["fact", "verif"], "fact_checking"),
        (["complete", "fill in"], "completion"),
    ]

    for keywords, family in keyword_map:
        for kw in keywords:
            if kw in defn_lower:
                return family

    # Fall back to name-based inference
    return get_task_family(task_name)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scaled TaskMap Training (50+ SNI tasks)")

    # Model
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--config", type=str, default="configs/taskmap_reference.yaml")

    # Training
    parser.add_argument("--max_steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/taskmap_scaled_50")
    parser.add_argument("--microbatch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)

    # TaskMap architecture overrides
    parser.add_argument("--active_fraction", type=float, default=None)
    parser.add_argument("--code_dim", type=int, default=None)
    parser.add_argument("--unfreeze_mapper", action="store_true")
    parser.add_argument("--mapping_loss", action="store_true")
    parser.add_argument("--shared_projector", action="store_true")
    parser.add_argument("--global_code", action="store_true")

    # Data
    parser.add_argument("--max_per_task", type=int, default=2000,
                        help="Max examples per task from SNI")
    parser.add_argument("--sni_cache_dir", type=str, default=None,
                        help="Cache directory for SNI dataset")
    parser.add_argument("--max_eval_examples", type=int, default=200,
                        help="Max evaluation examples per task")

    # Eval
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=3000)

    # Debug
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
    if args.active_fraction is not None:
        cfg["active_fraction"] = args.active_fraction
    if args.code_dim is not None:
        cfg["code_dim"] = args.code_dim
    if args.microbatch_size is not None:
        cfg["microbatch_size"] = args.microbatch_size
    if args.gradient_accumulation_steps is not None:
        cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.max_seq_length is not None:
        cfg["max_seq_length"] = args.max_seq_length
    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sni_data(task_list, split, max_per_task, cache_dir=None):
    """Load SNI tasks, validate, format, and return task data dict.

    Returns:
        task_data: {task_name: [formatted_examples]}
        task_definitions: {task_name: definition_string}
        task_families: {task_name: family_string}
    """
    print(f"\nLoading SNI dataset ({len(task_list)} requested tasks, split={split})...")
    full_ds = load_sni_dataset(cache_dir)

    # Validate
    valid_tasks, missing_tasks = validate_sni_tasks(full_ds, task_list)
    if missing_tasks:
        print(f"  WARNING: {len(missing_tasks)} tasks not found in dataset:")
        for t in missing_tasks:
            print(f"    - {t}")
    print(f"  Valid tasks: {len(valid_tasks)}/{len(task_list)}")

    # Filter and format
    raw_data = filter_sni_tasks(full_ds, valid_tasks, max_per_task)

    task_data = {}
    task_definitions = {}
    task_families = {}
    task_metrics = {}

    for task_name in valid_tasks:
        if task_name not in raw_data or not raw_data[task_name]:
            print(f"  {task_name}: no examples after filtering, skipping")
            continue

        # Extract the definition from the first example
        definition = raw_data[task_name][0].get("definition", "")
        task_definitions[task_name] = definition

        # Infer family from definition (more accurate than name-only)
        family = infer_family_from_definition(task_name, definition)
        task_families[task_name] = family

        # Format examples
        formatted = format_sni_examples(task_name, raw_data[task_name], split)
        if not formatted:
            print(f"  {task_name}: no examples after formatting, skipping")
            continue

        task_data[task_name] = formatted

        # Infer metric
        metric, max_tokens = infer_metric(task_name, formatted)
        task_metrics[task_name] = {"metric": metric, "max_response_tokens": max_tokens}

        print(f"  {task_name}: {len(formatted)} examples, family={family}, "
              f"metric={metric}")

    print(f"\nLoaded {len(task_data)} tasks with "
          f"{sum(len(v) for v in task_data.values()):,} total examples")

    return task_data, task_definitions, task_families, task_metrics


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_taskmap_scaled(cfg, backbone_model, tokenizer, task_ids,
                         task_definitions, device,
                         unfreeze_mapper=False, shared_projector=False,
                         global_code=False):
    """Initialize TaskMap for scaled SNI experiment."""
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

    freeze_mapper = not unfreeze_mapper
    taskmap = TaskMapModel(
        tm_config, num_tasks=len(task_ids),
        freeze_mapper=freeze_mapper,
        shared_projector=shared_projector,
        global_code=global_code,
    ).to(device)

    print(f"  Mapper: {'TRAINABLE' if unfreeze_mapper else 'frozen'}")
    if shared_projector:
        print(f"  Projector: SHARED across {tm_config.num_layers} layers")
    if global_code:
        print(f"  Task codes: GLOBAL (one per task, not per task-layer)")

    taskmap.register_tasks(task_ids)

    # Compute description embeddings from SNI definitions
    print(f"Computing description embeddings for {len(task_ids)} tasks...")
    for tid in task_ids:
        definition = task_definitions.get(tid, f"Complete the following task: {tid}")
        # Truncate long definitions for embedding
        desc = definition[:200] if definition else f"Complete the following task: {tid}"
        embed = taskmap.task_code.compute_description_embedding(
            backbone_model, tokenizer, desc, device
        )
        taskmap.cache_description(tid, embed)
        if len(task_ids) <= 60:  # print individual tasks for manageable counts
            print(f"  {tid}: '{desc[:50]}...' -> norm={embed.norm():.3f}")

    print(f"  All {len(task_ids)} embeddings cached")

    # Install FFN hooks
    hook_manager = TaskMapHookManager(
        backbone_model, taskmap, block_size=cfg.get("block_size", 128)
    )

    return taskmap, tm_config, hook_manager


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sni_tasks(backbone_model, tokenizer, taskmap, hook_manager,
                       eval_data, task_metrics, device, max_examples=200,
                       label="Eval"):
    """Evaluate a set of SNI tasks and return per-task and macro scores."""
    print(f"\n=== {label}: {len(eval_data)} tasks ===")
    backbone_model.eval()
    all_scores = {}

    for tid, examples in eval_data.items():
        if tid not in task_metrics:
            print(f"  Skipping {tid}: no metric info")
            continue

        # Truncate to max_examples
        eval_examples = examples[:max_examples]
        metric_name = task_metrics[tid]["metric"]
        max_tokens = task_metrics[tid]["max_response_tokens"]

        # Activate TaskMap hooks for this task
        try:
            taskmap.clear_route_cache()
            taskmap.compute_route(tid, device)
            hook_manager.activate_for_task(tid, device)
        except Exception as e:
            print(f"  {tid}: route activation failed: {e}")
            continue

        # Generate predictions
        try:
            predictions = generate_predictions(
                backbone_model, tokenizer, eval_examples,
                max_new_tokens=max_tokens, device=device,
            )
            references = [ex["response"] for ex in eval_examples]

            metric_fn = METRIC_FNS.get(metric_name)
            if metric_fn is None:
                # Fall back to accuracy or rouge_l
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

    return all_scores


@torch.no_grad()
def evaluate_coldstart_sni(backbone_model, tokenizer, taskmap, hook_manager,
                           holdout_data, holdout_definitions, holdout_metrics,
                           device, max_examples=200):
    """Cold-start evaluation: description-only routing for held-out tasks.

    These tasks were never trained, so the task codes are zero-initialized.
    Routing comes entirely from the description embedding.
    """
    print(f"\n=== Cold-Start Evaluation ({len(holdout_data)} held-out tasks) ===")
    backbone_model.eval()
    all_scores = {}

    for tid, examples in holdout_data.items():
        if tid not in holdout_metrics:
            continue

        # Register the held-out task and cache its description embedding
        desc = holdout_definitions.get(tid, f"Complete the following task: {tid}")
        desc = desc[:200]
        try:
            embed = taskmap.task_code.compute_description_embedding(
                backbone_model, tokenizer, desc, device
            )
            taskmap.cache_description(tid, embed)
        except Exception as e:
            print(f"  Skipping {tid}: embedding failed: {e}")
            continue

        # Compute route from description only (residual codes are zero/absent)
        try:
            taskmap.clear_route_cache()
            taskmap.compute_route(tid, device)
            hook_manager.activate_for_task(tid, device)
        except Exception as e:
            print(f"  Skipping {tid}: route failed: {e}")
            continue

        eval_examples = examples[:max_examples]
        metric_name = holdout_metrics[tid]["metric"]
        max_tokens = holdout_metrics[tid]["max_response_tokens"]

        try:
            predictions = generate_predictions(
                backbone_model, tokenizer, eval_examples,
                max_new_tokens=max_tokens, device=device,
            )
            references = [ex["response"] for ex in eval_examples]

            metric_fn = METRIC_FNS.get(metric_name, accuracy)
            scores = metric_fn(predictions, references)
            all_scores[tid] = scores
            primary = list(scores.values())[0]
            family = infer_family_from_definition(tid, holdout_definitions.get(tid, ""))
            print(f"  {tid}: {primary:.2f} ({metric_name}, family={family})")

        except Exception as e:
            print(f"  Skipping {tid} eval: {e}")
            continue

    if all_scores:
        primary_values = [list(v.values())[0] for v in all_scores.values()]
        macro = float(np.mean(primary_values))
        all_scores["macro_avg"] = macro
        print(f"\n  Cold-start macro: {macro:.2f} "
              f"(over {len(primary_values)} tasks)")
    else:
        print("  No cold-start tasks evaluated successfully")

    return all_scores


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_taskmap_scaled(args):
    cfg = load_config(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = cfg.get("output_dir", "outputs/taskmap_scaled_50")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load backbone (frozen) ──
    backbone_name = cfg.get("backbone", "Qwen/Qwen2.5-1.5B")
    print(f"Loading frozen backbone: {backbone_name}")
    backbone_model, tokenizer = load_backbone(backbone_name)
    backbone_model = backbone_model.to(device)
    backbone_model.eval()
    for p in backbone_model.parameters():
        p.requires_grad = False

    # ── Load SNI training data ──
    # Remove duplicates from TRAIN_TASKS_SNI (task391 appears twice)
    seen = set()
    train_task_list = []
    for t in TRAIN_TASKS_SNI:
        if t not in seen:
            seen.add(t)
            train_task_list.append(t)

    train_data, train_definitions, train_families, train_metrics = load_sni_data(
        train_task_list, split="train",
        max_per_task=args.max_per_task, cache_dir=args.sni_cache_dir,
    )
    task_ids = list(train_data.keys())
    total_examples = sum(len(v) for v in train_data.values())
    print(f"\nTraining tasks: {len(task_ids)}")
    print(f"Total training examples: {total_examples:,}")
    print(f"Families represented: "
          f"{sorted(set(train_families.values()))}")

    if len(task_ids) == 0:
        print("ERROR: No valid training tasks. Exiting.")
        sys.exit(1)

    # ── Setup TaskMap ──
    print("\nSetting up TaskMap model...")
    taskmap, tm_config, hook_manager = setup_taskmap_scaled(
        cfg, backbone_model, tokenizer, task_ids,
        train_definitions, device,
        unfreeze_mapper=args.unfreeze_mapper,
        shared_projector=args.shared_projector,
        global_code=args.global_code,
    )
    summary = taskmap.parameter_summary()
    print(f"\nTaskMap parameter summary: {summary}")

    # ── Build family pairs for topology loss ──
    family_pairs = build_family_pairs_from_map(task_ids, train_families)
    print(f"Family pairs for topology loss: {len(family_pairs)}")

    # ── Optimizer (separate param groups) ──
    param_groups = []
    code_params = []
    projector_params = []
    for name, param in taskmap.task_code.named_parameters():
        if param.requires_grad:
            if "projector" in name:
                projector_params.append(param)
            else:
                code_params.append(param)
    param_groups.append({"params": code_params,
                         "lr": cfg.get("code_learning_rate", 2e-3)})
    param_groups.append({"params": projector_params,
                         "lr": cfg.get("projector_learning_rate", 2e-4)})

    if args.unfreeze_mapper:
        mapper_params = [p for p in taskmap.mapper_bank.parameters()
                         if p.requires_grad]
        param_groups.append({"params": mapper_params,
                             "lr": cfg.get("projector_learning_rate", 2e-4)})
        print(f"  Mapper params in optimizer: "
              f"{sum(p.numel() for p in mapper_params):,}")

    optimizer = AdamW(param_groups,
                      weight_decay=cfg.get("weight_decay", 0.01),
                      betas=(0.9, 0.95))

    max_steps = 2 if args.dry_run else cfg.get("max_steps", 12000)
    warmup_steps = int(max_steps * cfg.get("warmup_fraction", 0.03))
    warmup_sched = LinearLR(optimizer, start_factor=0.01,
                            total_iters=max(warmup_steps, 1))
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=max(max_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    # ── Loss computer ──
    use_mapping_loss = args.mapping_loss
    if use_mapping_loss:
        print("  Mapping Networks losses ACTIVE in backward pass")
    loss_computer = TaskMapLossComputer(
        tm_config, family_pairs, train_families,
        lambda_bud=cfg.get("lambda_bud", 0.05),
        lambda_topo=cfg.get("lambda_topo", 0.01),
        lambda_bal=cfg.get("lambda_bal", 0.01),
        lambda_stab=cfg.get("lambda_stab", 1e-3),
        lambda_sm=cfg.get("lambda_sm", 1e-3),
        lambda_align=cfg.get("lambda_align", 1e-4),
        active_mapping_loss=use_mapping_loss,
    )

    # ── Training loop ──
    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    max_seq = cfg.get("max_seq_length", 2048)
    microbatch_size = cfg.get("microbatch_size", 4)

    print(f"\nStarting scaled TaskMap training for {max_steps} steps...")
    print(f"  Tasks: {len(task_ids)}")
    print(f"  Warmup: {warmup_steps} steps (dense), then Gumbel anneal")
    print(f"  Active fraction: {tm_config.active_fraction} "
          f"({taskmap.router.k}/{tm_config.num_blocks} blocks)")
    print(f"  Microbatch: {microbatch_size}, grad accum: {grad_accum}")

    global_step = 0
    accum_loss = 0.0
    all_route_masks = {}
    t_start = time.time()

    dataloader = build_dataloader(train_data, microbatch_size,
                                  max_steps * grad_accum, args.seed)

    for step_idx, (task_id, examples) in enumerate(dataloader):
        # ── Compute route fresh each microbatch ──
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
                loss_str = " | ".join(
                    f"{k}={v.item():.4f}" if torch.is_tensor(v)
                    else f"{k}={v:.4f}"
                    for k, v in loss_dict.items() if k != "total"
                )
                family = train_families.get(task_id, "?")
                print(f"  Step {global_step}/{max_steps} | "
                      f"Total: {accum_loss:.4f} | {loss_str} | "
                      f"Task: {task_id} ({family}) | Phase: {phase} | "
                      f"Time: {elapsed:.0f}s")
                accum_loss = 0.0

            # Periodic checkpoint
            if (not args.dry_run
                    and global_step % args.save_every == 0):
                save_path = os.path.join(output_dir,
                                         f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    "task_code_state": taskmap.task_code.state_dict(),
                    "step": global_step,
                    "config": cfg,
                    "task_ids": task_ids,
                    "task_families": train_families,
                    "task_metrics": train_metrics,
                }, os.path.join(save_path, "taskmap_state.pt"))
                print(f"  Saved checkpoint to {save_path}")

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({global_step} steps)")

    # ── Save final model ──
    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    torch.save({
        "task_code_state": taskmap.task_code.state_dict(),
        "mapper_state": taskmap.mapper_bank.state_dict(),
        "config": cfg,
        "step": global_step,
        "task_ids": task_ids,
        "task_families": train_families,
        "task_metrics": train_metrics,
        "task_definitions": train_definitions,
    }, os.path.join(final_path, "taskmap_state.pt"))
    print(f"Saved final model to {final_path}")

    # ==================================================================
    # Evaluation Phase
    # ==================================================================

    # ── 1. Evaluate on trained tasks (validation split) ──
    print("\n" + "=" * 70)
    print("EVALUATION PHASE")
    print("=" * 70)

    # Load validation split for training tasks
    val_data, _, _, val_metrics = load_sni_data(
        task_ids, split="validation",
        max_per_task=args.max_per_task, cache_dir=args.sni_cache_dir,
    )
    # Merge metrics (validation split may miss some)
    for tid in val_data:
        if tid not in val_metrics:
            val_metrics[tid] = train_metrics.get(tid, {"metric": "accuracy",
                                                       "max_response_tokens": 16})

    trained_scores = evaluate_sni_tasks(
        backbone_model, tokenizer, taskmap, hook_manager,
        val_data, val_metrics, device,
        max_examples=args.max_eval_examples,
        label="Trained Tasks (validation)",
    )

    # ── 2. Route analysis ──
    print("\n=== Route Analysis ===")
    from analysis.route_overlap import (compute_route_overlaps,
                                        print_route_report)

    overlaps, within_avg, between_avg = compute_route_overlaps(
        taskmap, task_ids, train_families, device
    )
    print_route_report(overlaps, within_avg, between_avg, train_families)

    # Per-layer within vs between
    print("\n--- Per-layer within vs between family overlap ---")
    num_layers = taskmap.config.num_layers
    for l in range(num_layers):
        within_l = []
        between_l = []
        for (t1, t2), layer_overlaps in overlaps.items():
            if train_families.get(t1) == train_families.get(t2):
                within_l.append(layer_overlaps[l])
            else:
                between_l.append(layer_overlaps[l])
        w = np.mean(within_l) if within_l else 0
        b = np.mean(between_l) if between_l else 0
        print(f"  Layer {l:2d}: within={w:.3f}  between={b:.3f}  "
              f"ratio={w / max(b, 1e-8):.2f}x")

    # ── 3. Cold-start evaluation on held-out tasks ──
    # Remove any holdout task that was already in training
    holdout_list = [t for t in HOLDOUT_TASKS_SNI if t not in set(task_ids)]
    print(f"\nHeld-out task list: {len(holdout_list)} tasks "
          f"(removed {len(HOLDOUT_TASKS_SNI) - len(holdout_list)} overlapping)")

    holdout_data, holdout_defs, holdout_families, holdout_metrics = load_sni_data(
        holdout_list, split="validation",
        max_per_task=args.max_per_task, cache_dir=args.sni_cache_dir,
    )

    # Register holdout tasks with zero-initialized codes
    for tid in holdout_data:
        if tid not in task_ids:
            try:
                taskmap.register_tasks([tid])
            except Exception:
                pass  # may already be registered or max capacity

    coldstart_scores = evaluate_coldstart_sni(
        backbone_model, tokenizer, taskmap, hook_manager,
        holdout_data, holdout_defs, holdout_metrics,
        device, max_examples=args.max_eval_examples,
    )

    # ==================================================================
    # Final results summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Per-family breakdown for trained tasks
    if trained_scores:
        family_scores = defaultdict(list)
        for tid, scores in trained_scores.items():
            if tid == "macro_avg":
                continue
            fam = train_families.get(tid, "other")
            family_scores[fam].append(list(scores.values())[0])

        print("\nTrained tasks - per family:")
        for fam in sorted(family_scores.keys()):
            vals = family_scores[fam]
            print(f"  {fam:20s}: {np.mean(vals):.2f} "
                  f"(n={len(vals)}, range=[{min(vals):.1f}, {max(vals):.1f}])")
        print(f"  {'MACRO':20s}: {trained_scores.get('macro_avg', 0):.2f}")

    # Per-family breakdown for cold-start tasks
    if coldstart_scores:
        cs_family_scores = defaultdict(list)
        for tid, scores in coldstart_scores.items():
            if tid == "macro_avg":
                continue
            fam = holdout_families.get(tid, "other")
            cs_family_scores[fam].append(list(scores.values())[0])

        print("\nCold-start tasks - per family:")
        for fam in sorted(cs_family_scores.keys()):
            vals = cs_family_scores[fam]
            print(f"  {fam:20s}: {np.mean(vals):.2f} "
                  f"(n={len(vals)})")
        print(f"  {'MACRO':20s}: {coldstart_scores.get('macro_avg', 0):.2f}")

    # Build full results dict
    active_frac = cfg.get("active_fraction", 0.50)
    results = {
        "experiment": "taskmap_scaled_50_sni",
        "backbone": backbone_name,
        "num_train_tasks": len(task_ids),
        "num_holdout_tasks": len(holdout_data),
        "max_steps": max_steps,
        "active_fraction": active_frac,
        "trained_scores": trained_scores,
        "route_analysis": {
            "within_family_overlap": float(within_avg),
            "between_family_overlap": float(between_avg),
            "ratio": float(within_avg / max(between_avg, 1e-8)),
        },
        "cold_start_scores": coldstart_scores,
        "task_ids": task_ids,
        "task_families": train_families,
        "training_time_seconds": total_time,
    }

    # Write results JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {results_path}")

    # Also print to stdout for pod log scraping
    print("\n=== RESULTS JSON ===")
    print(json.dumps(results, indent=2, default=str))
    print("=== END RESULTS ===")

    hook_manager.remove_all()
    return taskmap, backbone_model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    train_taskmap_scaled(args)
