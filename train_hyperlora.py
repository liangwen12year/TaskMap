"""
Description-conditioned LoRA generator baseline (Hypter/LoRAGen-style).

A small MLP takes a task description embedding and generates LoRA A,B factors
for each layer's FFN projections. Unlike TaskMap which generates coefficients
for fixed orthogonal bases, this generates full low-rank factors directly.

To keep the output dimension tractable, the generator produces a compact code
per layer, which is then projected to LoRA shapes via fixed random matrices.

Usage:
  python train_hyperlora.py --backbone Qwen/Qwen2.5-1.5B --max_steps 6000
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

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone, count_parameters
from models.task_code import TaskCodeModule
from data.config import KNOWN_TASKS
from data.download import download_task
from data.format import format_all_tasks
from data.sampler import build_dataloader
from train import tokenize_batch, set_seed
from eval import evaluate_task


class HyperLoRAGenerator(nn.Module):
    """Generates LoRA A,B factors from task description embedding."""

    def __init__(self, embed_dim, num_layers, rank, model_dim, ffn_dim,
                 hidden_dim=512, code_dim=128):
        super().__init__()
        self.num_layers = num_layers
        self.rank = rank
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim

        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.layer_heads = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_heads.append(nn.Linear(hidden_dim, code_dim))

        a_up_size = model_dim * rank
        b_up_size = rank * ffn_dim
        a_down_size = ffn_dim * rank
        b_down_size = rank * model_dim
        total_per_layer = 2 * (a_up_size + b_up_size) + (a_down_size + b_down_size)

        self.proj_seeds = {}
        for l in range(num_layers):
            g = torch.Generator().manual_seed(42 + l)
            proj = torch.randn(code_dim, total_per_layer, generator=g) * 0.01
            self.register_buffer(f"proj_l{l}", proj)

    def forward(self, embedding, layer_idx):
        h = self.encoder(embedding)
        code = self.layer_heads[layer_idx](h)
        proj = getattr(self, f"proj_l{layer_idx}")
        flat = code @ proj

        offset = 0
        factors = {}
        for name, out_d, in_d in [
            ("up_A", self.model_dim, self.rank),
            ("up_B", self.rank, self.ffn_dim),
            ("gate_A", self.model_dim, self.rank),
            ("gate_B", self.rank, self.ffn_dim),
            ("down_A", self.ffn_dim, self.rank),
            ("down_B", self.rank, self.model_dim),
        ]:
            size = out_d * in_d
            factors[name] = flat[offset:offset+size].view(out_d, in_d)
            offset += size

        return factors


class HyperLoRAHook:
    """Applies generated LoRA factors to one FFN layer."""

    def __init__(self, layer_module, layer_idx, model_dim, ffn_dim):
        self.layer_idx = layer_idx
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.factors = None
        self.handle = layer_module.register_forward_hook(self.hook_fn)

    def set_factors(self, factors):
        self.factors = factors

    def hook_fn(self, module, args, output):
        if self.factors is None:
            return output

        h = args[0] if isinstance(args, tuple) else args
        if h.dim() == 3:
            h_2d = h.view(-1, h.size(-1))
        else:
            h_2d = h

        dtype = h_2d.dtype
        f = {k: v.to(dtype) for k, v in self.factors.items()}
        delta_up = h_2d @ f["up_A"] @ f["up_B"]
        delta_gate = h_2d @ f["gate_A"] @ f["gate_B"]

        up_proj = module.up_proj(h_2d) if hasattr(module, 'up_proj') else None
        gate_proj = module.gate_proj(h_2d) if hasattr(module, 'gate_proj') else None

        if up_proj is not None and gate_proj is not None:
            act_orig = F.silu(gate_proj) * up_proj
            act_mod = F.silu(gate_proj + delta_gate) * (up_proj + delta_up)
            delta_act = act_mod - act_orig

            down_orig = module.down_proj.weight.to(dtype)
            delta_down = f["down_A"] @ f["down_B"]
            delta_out = delta_act @ (down_orig + delta_down).T - delta_act @ down_orig.T + act_orig @ delta_down.T

            if h.dim() == 3:
                delta_out = delta_out.view(h.size(0), h.size(1), -1)
            return output + delta_out

        return output

    def remove(self):
        self.handle.remove()


def parse_args():
    parser = argparse.ArgumentParser(description="HyperLoRA Training")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--code_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="outputs/hyperlora")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def train_hyperlora(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

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

    generator = HyperLoRAGenerator(
        embed_dim=embed_dim, num_layers=num_layers, rank=args.rank,
        model_dim=model_dim, ffn_dim=ffn_dim,
        hidden_dim=args.hidden_dim, code_dim=args.code_dim,
    ).to(device)

    trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"HyperLoRA generator: {trainable:,} trainable parameters")

    task_code = TaskCodeModule(
        num_layers=1, embed_dim=embed_dim, code_dim=embed_dim, num_tasks=0
    )

    task_ids = [t for t in KNOWN_TASKS.keys()
                if KNOWN_TASKS[t]["metric"] != "pass_at_1"]

    print(f"Computing description embeddings for {len(task_ids)} tasks...")
    task_embeds = {}
    for tid in task_ids:
        desc = KNOWN_TASKS[tid]["descriptions"][0]
        embed = task_code.compute_description_embedding(
            backbone, tokenizer, desc, device
        )
        task_embeds[tid] = embed
        print(f"  {tid}: norm={embed.norm():.3f}")

    if hasattr(backbone, 'model'):
        mlp_layers = [backbone.model.layers[i].mlp for i in range(num_layers)]
    else:
        mlp_layers = [backbone.transformer.h[i].mlp for i in range(num_layers)]

    hooks = []
    for l in range(num_layers):
        hook = HyperLoRAHook(mlp_layers[l], l, model_dim, ffn_dim)
        hooks.append(hook)

    print("Loading training data...")
    datasets = {}
    for tid in task_ids:
        meta = KNOWN_TASKS[tid]
        ds = download_task(tid, meta)
        if ds is not None:
            datasets[tid] = ds

    train_data = format_all_tasks(datasets, split="train")
    total_examples = sum(len(v) for v in train_data.values())
    print(f"Total training examples: {total_examples:,}")

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

    print(f"\nStarting HyperLoRA training for {max_steps} steps...")
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
                print(f"  Step {global_step}/{max_steps} | "
                      f"Loss: {accum_loss:.4f} | "
                      f"Task: {task_id} | Time: {elapsed:.0f}s")
                accum_loss = 0.0

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s")

    # Evaluation
    print("\n=== Evaluation ===")
    generator.eval()
    eval_data = format_all_tasks(datasets, split="validation")

    max_eval = int(os.environ.get("TASKMAP_EVAL_EXAMPLES", "500"))
    all_scores = {}

    for tid in task_ids:
        if tid not in eval_data:
            continue
        examples = eval_data[tid][:max_eval]
        meta = KNOWN_TASKS[tid]

        embed = task_embeds[tid]
        with torch.no_grad():
            for l in range(num_layers):
                factors = generator(embed, l)
                hooks[l].set_factors(factors)

        scores = evaluate_task(
            backbone, tokenizer, tid, examples,
            meta["metric"], meta["max_response_tokens"], device
        )
        all_scores[tid] = scores
        primary = list(scores.values())[0]
        print(f"    {tid}: {scores}")

    if all_scores:
        primary_values = [list(v.values())[0] for v in all_scores.values()]
        macro = float(np.mean(primary_values))
        all_scores["macro_avg"] = macro
        print(f"\n  Macro: {macro:.2f}")

    for hook in hooks:
        hook.set_factors(None)

    results = {
        "mode": "hyperlora",
        "backbone": args.backbone,
        "rank": args.rank,
        "trainable_params": trainable,
        "scores": all_scores,
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== RESULTS JSON ===")
    print(json.dumps(results, indent=2, default=str))
    print("=== END RESULTS ===")
    print("=== DONE ===")


if __name__ == "__main__":
    args = parse_args()
    train_hyperlora(args)
