"""
Evaluation runner for all methods (Table 3).

Loads a trained model (baseline LoRA or TaskMap), evaluates on all tasks,
computes per-task metrics, macro average, and negative transfer.

Usage:
  # Evaluate frozen base
  python run_eval.py --mode frozen --backbone Qwen/Qwen2.5-0.5B

  # Evaluate LoRA checkpoint
  python run_eval.py --mode lora --checkpoint outputs/lora_r16/final

  # Evaluate TaskMap checkpoint
  python run_eval.py --mode taskmap --checkpoint outputs/taskmap/final

  # Dry run (2 tasks only)
  python run_eval.py --mode frozen --backbone Qwen/Qwen2.5-0.5B --dry_run
"""

import os
import sys
import json
import argparse
import torch
import time

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import load_backbone, add_lora, count_parameters
from data.config import KNOWN_TASKS
from data.download import download_task
from data.format import format_all_tasks
from eval import evaluate_all, compute_negative_transfer


def parse_args():
    parser = argparse.ArgumentParser(description="TaskMap Evaluation")
    parser.add_argument("--mode", choices=["frozen", "lora", "taskmap"], default="frozen")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--max_examples", type=int, default=200,
                        help="Max examples per task for eval (reduce for speed)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Evaluate only 2 tasks with 10 examples each")
    return parser.parse_args()


def load_model(args):
    """Load model based on mode and checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading backbone: {args.backbone}")
    model, tokenizer = load_backbone(args.backbone)

    if args.mode == "lora":
        if args.checkpoint:
            from peft import PeftModel
            print(f"Loading LoRA from checkpoint: {args.checkpoint}")
            model = PeftModel.from_pretrained(model, args.checkpoint)
        else:
            model = add_lora(model, rank=args.lora_rank)

    model = model.to(device)
    model.eval()
    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    return model, tokenizer, device


def load_eval_data(args):
    """Load and format evaluation data."""
    datasets = {}
    task_configs = {}

    tasks_to_eval = list(KNOWN_TASKS.keys())
    if args.dry_run:
        tasks_to_eval = tasks_to_eval[:2]

    for task_id in tasks_to_eval:
        meta = KNOWN_TASKS[task_id]
        ds = download_task(task_id, meta)
        if ds is not None:
            datasets[task_id] = ds
            task_configs[task_id] = meta

    eval_data = format_all_tasks(datasets, split="validation")

    # Limit examples per task
    max_ex = 10 if args.dry_run else args.max_examples
    for task_id in eval_data:
        if len(eval_data[task_id]) > max_ex:
            eval_data[task_id] = eval_data[task_id][:max_ex]

    return eval_data, task_configs


def measure_efficiency(model, tokenizer, device, seq_length=2048):
    """
    Measure throughput and peak memory (Table 3 columns).

    Reports:
    - Tokens/s: median of 100 steps after 20 warmup
    - Peak GB: peak allocated GPU memory
    """
    if device != "cuda":
        return {"tokens_per_sec": 0, "peak_gb": 0, "note": "CPU mode, skipped"}

    model.eval()
    dummy_input = tokenizer("Hello world " * 100, return_tensors="pt",
                             truncation=True, max_length=seq_length).to(device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            model(**dummy_input)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**dummy_input)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    import numpy as np
    median_time = np.median(times)
    tokens = dummy_input["input_ids"].shape[1]
    tokens_per_sec = tokens / median_time
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    return {"tokens_per_sec": tokens_per_sec, "peak_gb": peak_gb}


def run_evaluation(args):
    model, tokenizer, device = load_model(args)

    print("\nLoading evaluation data...")
    eval_data, task_configs = load_eval_data(args)
    print(f"Evaluating {len(eval_data)} tasks")

    print("\nRunning evaluation...")
    scores = evaluate_all(model, tokenizer, eval_data, task_configs, device)

    # Efficiency
    print("\nMeasuring efficiency...")
    efficiency = measure_efficiency(model, tokenizer, device)
    scores["efficiency"] = efficiency

    # Parameter counts
    trainable, total = count_parameters(model)
    scores["params"] = {
        "trainable_M": trainable / 1e6,
        "total_M": total / 1e6,
    }

    # Save results
    output = {
        "mode": args.mode,
        "backbone": args.backbone,
        "checkpoint": args.checkpoint,
        "scores": {k: (v if not isinstance(v, dict) else
                       {kk: float(vv) if hasattr(vv, 'item') else vv
                        for kk, vv in v.items()})
                   for k, v in scores.items()},
    }

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_file}")

    # Also print results to stdout for pod log capture
    print("\n=== RESULTS JSON ===")
    print(json.dumps(output, indent=2, default=str))
    print("=== END RESULTS ===")

    return scores


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
