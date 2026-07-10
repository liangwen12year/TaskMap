"""
Training script for baseline methods (frozen base, dense multi-task LoRA).

Paper Section 4.2 (Tier A):
- Qwen2.5-1.5B or Llama-3.2-1B, frozen decoder-only backbone
- BF16, FlashAttention, seq length 2048
- Global batch of 65,536 tokens
- 12,000 optimizer steps
- AdamW (beta1=0.9, beta2=0.95), weight decay 0.01
- Cosine decay LR schedule, 3% warmup

Usage:
  # Frozen base (no training, just eval)
  python train.py --mode frozen --backbone Qwen/Qwen2.5-1.5B

  # Dense multi-task LoRA
  python train.py --mode lora --backbone Qwen/Qwen2.5-1.5B --lora_rank 16

  # Dry run (2 steps, no GPU needed for testing)
  python train.py --mode lora --dry_run
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

from models.backbone import load_backbone, add_lora, count_parameters
from data.config import KNOWN_TASKS
from data.download import download_task
from data.format import format_dataset, format_all_tasks
from data.sampler import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="TaskMap Baseline Training")
    parser.add_argument("--mode", choices=["frozen", "lora"], default="lora")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="outputs/baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Run 2 steps for testing")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_batch(tokenizer, examples, max_length=2048):
    """Tokenize a microbatch, compute loss only on the response span."""
    texts = [ex["full_text"] for ex in examples]

    encodings = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()

    for i, ex in enumerate(examples):
        sep = "Response:" if "Response:" in ex["full_text"] else "Output:"
        prompt_part = ex["full_text"].split(sep)[0] + sep
        prompt_ids = tokenizer(prompt_part, truncation=True, max_length=max_length)["input_ids"]
        prompt_len = len(prompt_ids)
        labels[i, :prompt_len] = -100

    labels[attention_mask == 0] = -100

    # Safety: if all labels are -100 for any example, keep last token as target
    for i in range(len(examples)):
        if (labels[i] != -100).sum() == 0:
            last_real = attention_mask[i].sum().item() - 1
            if last_real >= 0:
                labels[i, last_real] = input_ids[i, last_real]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    print(f"Loading backbone: {args.backbone}")
    model, tokenizer = load_backbone(args.backbone)

    if args.mode == "lora":
        print(f"Adding LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
        model = add_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    model = model.to(device)
    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    if args.mode == "frozen":
        print("Frozen mode: skipping training, proceeding to eval.")
        return model, tokenizer

    # ── Load data ──
    print("\nLoading training data...")
    datasets = {}
    for task_id, meta in KNOWN_TASKS.items():
        ds = download_task(task_id, meta)
        if ds is not None:
            datasets[task_id] = ds

    print("\nFormatting datasets...")
    train_data = format_all_tasks(datasets, split="train")
    total_examples = sum(len(v) for v in train_data.values())
    print(f"Total training examples: {total_examples:,}")

    # ── Optimizer ──
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    warmup_steps = int(args.max_steps * args.warmup_ratio)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_steps])

    max_steps = 2 if args.dry_run else args.max_steps

    # ── Training loop ──
    print(f"\nStarting training for {max_steps} steps...")
    model.train()
    global_step = 0
    accum_loss = 0.0
    t_start = time.time()

    dataloader = build_dataloader(train_data, args.microbatch_size, max_steps * args.gradient_accumulation_steps, args.seed)

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
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step}/{max_steps} | "
                      f"Loss: {accum_loss:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Task: {task_id} | "
                      f"Time: {elapsed:.0f}s")
                accum_loss = 0.0

            if not args.dry_run and global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(save_path)
                print(f"  Saved checkpoint to {save_path}")

            if global_step >= max_steps:
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({global_step} steps)")

    # Save final
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    print(f"Saved final model to {final_path}")

    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    model, tokenizer = train(args)
    print("\nDone. Run eval.py separately to evaluate.")
