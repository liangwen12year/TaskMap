"""
Direct Block-LoRA training (Table 3, row 7).

Same block routes and low-rank residual shapes as TaskMap, but route logits
and coefficients are directly optimized per task (no mapper). This isolates
whether the Mapping Networks contribution matters, or if the routing
mechanism itself is the issue.

Usage:
  python train_direct_block_lora.py --backbone Qwen/Qwen2.5-1.5B --max_steps 6000
  python train_direct_block_lora.py --dry_run
"""

import os
import sys
import time
import argparse
import json
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone, count_parameters
from models.taskmap_model import TaskMapConfig
from models.baselines import DirectBlockLoRA
from models.block_residuals import BlockResidualBases
from models.ffn_hooks import TaskMapFFNHook, TaskMapHookManager
from data.config import KNOWN_TASKS
from data.download import download_task
from data.format import format_all_tasks
from data.sampler import build_dataloader
from train import tokenize_batch, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Direct Block-LoRA Training")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--active_fraction", type=float, default=0.5)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--output_dir", type=str, default="outputs/direct_block_lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


class DirectBlockLoRAHookManager:
    """
    Hooks for Direct Block-LoRA: uses directly optimized routes and
    coefficients instead of mapper-generated ones.
    """

    def __init__(self, backbone_model, dbl_model, residual_bases, config):
        self.backbone = backbone_model
        self.dbl = dbl_model
        self.bases = residual_bases
        self.config = config
        self.hooks = []
        self._install_hooks()

    def _get_mlp_layers(self):
        if hasattr(self.backbone, 'model'):
            model = self.backbone.model
        elif hasattr(self.backbone, 'base_model'):
            model = self.backbone.base_model.model
        else:
            model = self.backbone
        if hasattr(model, 'layers'):
            return [layer.mlp for layer in model.layers]
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return [layer.mlp for layer in model.model.layers]
        return []

    def _install_hooks(self):
        mlp_layers = self._get_mlp_layers()
        for idx, mlp in enumerate(mlp_layers):
            hook = TaskMapFFNHook(idx, self.config.block_size)
            hook.register(mlp)
            self.hooks.append(hook)
        print(f"Installed Direct Block-LoRA hooks on {len(self.hooks)} MLP layers")

    def activate_for_task(self, task_id: str, device: str = "cuda"):
        for layer_idx, hook in enumerate(self.hooks):
            if layer_idx >= self.config.num_layers:
                break
            mask, selected, logits = self.dbl.get_route(task_id, layer_idx)
            c_u, c_g, c_d = self.dbl.get_coefficients(task_id, layer_idx)
            hook.set_route({
                'mask': mask,
                'selected': selected,
                'c_u': c_u,
                'c_g': c_g,
                'c_d': c_d,
            }, self.bases)

    def deactivate(self):
        for hook in self.hooks:
            hook.deactivate()

    def remove_all(self):
        for hook in self.hooks:
            hook.remove()


def train_direct_block_lora(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load frozen backbone
    print(f"Loading frozen backbone: {args.backbone}")
    backbone, tokenizer = load_backbone(args.backbone)
    backbone = backbone.to(device)
    backbone.eval()
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

    # Setup config
    config = TaskMapConfig.from_backbone(
        args.backbone,
        block_size=args.block_size,
        active_fraction=args.active_fraction,
        rank=args.rank,
    )

    # Create Direct Block-LoRA model
    dbl = DirectBlockLoRA(config, task_ids).to(device)
    print(f"Direct Block-LoRA trainable params: {dbl.num_trainable():,}")

    # Create residual bases (frozen)
    bases = BlockResidualBases(
        config.num_layers, config.num_blocks, config.rank,
        config.model_dim, config.block_size
    ).to(device)

    # Install hooks
    hook_manager = DirectBlockLoRAHookManager(backbone, dbl, bases, config)

    # Optimizer — all DBL params are trainable
    optimizer = AdamW(dbl.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    max_steps = 2 if args.dry_run else args.max_steps
    warmup_steps = int(max_steps * args.warmup_ratio)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(max_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    # Training loop
    grad_accum = args.gradient_accumulation_steps
    print(f"\nStarting Direct Block-LoRA training for {max_steps} steps...")
    print(f"  Active fraction: {config.active_fraction} ({max(1, int(config.active_fraction * config.num_blocks))}/{config.num_blocks} blocks)")

    global_step = 0
    accum_loss = 0.0
    t_start = time.time()

    dataloader = build_dataloader(train_data, args.microbatch_size,
                                  max_steps * grad_accum, args.seed)

    for step_idx, (task_id, examples) in enumerate(dataloader):
        hook_manager.activate_for_task(task_id, device)

        batch = tokenize_batch(tokenizer, examples, args.max_seq_length)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = backbone(**batch)
        loss = outputs.loss / grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (step_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(dbl.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 100 == 0 or global_step == 1:
                avg_loss = accum_loss / (100 if global_step > 1 else 1)
                elapsed = time.time() - t_start
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step}/{max_steps} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                      f"Task: {task_id} | Time: {elapsed:.0f}s")
                accum_loss = 0.0

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({global_step} steps)")

    # Save
    torch.save(dbl.state_dict(), os.path.join(args.output_dir, "dbl_state.pt"))

    # Evaluate
    print("\n=== Starting Direct Block-LoRA Evaluation ===")
    eval_datasets = {}
    for tid, meta in KNOWN_TASKS.items():
        ds = download_task(tid, meta)
        if ds is not None:
            eval_datasets[tid] = ds
    eval_data = format_all_tasks(eval_datasets, split="validation")
    import os as _os
    max_eval = int(_os.environ.get("TASKMAP_EVAL_EXAMPLES", "500"))
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

    # ── Route Analysis ──
    print("\n=== Route Analysis ===")
    task_families_map = {tid: KNOWN_TASKS[tid]["family"] for tid in task_ids}

    # Compute Jaccard overlaps from DBL routes
    from analysis.route_overlap import jaccard_overlap
    route_masks = {}
    for tid in task_ids:
        masks = []
        for l in range(config.num_layers):
            mask, _, _ = dbl.get_route(tid, l)
            masks.append(mask.detach())
        route_masks[tid] = masks

    within_scores = []
    between_scores = []
    overlaps = {}
    for i, t1 in enumerate(task_ids):
        for t2 in task_ids[i + 1:]:
            layer_overlaps = []
            for l in range(config.num_layers):
                j = jaccard_overlap(route_masks[t1][l], route_masks[t2][l])
                layer_overlaps.append(j)
            overlaps[(t1, t2)] = layer_overlaps
            avg = np.mean(layer_overlaps)
            if task_families_map[t1] == task_families_map[t2]:
                within_scores.append(avg)
            else:
                between_scores.append(avg)

    within_avg = np.mean(within_scores) if within_scores else 0
    between_avg = np.mean(between_scores) if between_scores else 0
    print(f"Within-family overlap:  {within_avg:.3f}")
    print(f"Between-family overlap: {between_avg:.3f}")
    print(f"Ratio: {within_avg / max(between_avg, 1e-8):.2f}x")

    for (t1, t2), layer_overlaps in sorted(overlaps.items()):
        avg = np.mean(layer_overlaps)
        same = task_families_map[t1] == task_families_map[t2]
        print(f"  {t1:15s} <-> {t2:15s}: {avg:.3f}{' [SAME]' if same else ''}")

    print("\n--- Selected blocks per task (layer 0) ---")
    for tid in task_ids:
        _, selected, _ = dbl.get_route(tid, 0)
        print(f"  {tid:15s}: blocks {selected[:10]}{'...' if len(selected) > 10 else ''}")

    results = {
        "mode": "direct_block_lora",
        "scores": all_scores,
        "route_analysis": {
            "within_family_overlap": float(within_avg),
            "between_family_overlap": float(between_avg),
            "ratio": float(within_avg / max(between_avg, 1e-8)),
        }
    }
    print("\n=== RESULTS JSON ===")
    print(json.dumps(results, indent=2, default=str))
    print("=== END RESULTS ===")

    hook_manager.remove_all()


if __name__ == "__main__":
    args = parse_args()
    train_direct_block_lora(args)
