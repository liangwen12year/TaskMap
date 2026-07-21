"""
Route analysis for TaskMap (Professor Ge points 2 and 3).

Analyses:
1. Route entropy: how selective is each task's routing?
2. Block frequency: which blocks are selected most/least often?
3. Per-layer overlap heatmap data
4. Task-family routing patterns
5. Route similarity to nearest known task (for cold-start analysis)

Operates on saved route data from training runs, or computes routes
from a checkpoint.

Usage:
  python -m analysis.route_analysis \
    --results_file results/eval_taskmap_1.5b_500eval.json \
    --output_dir analysis/outputs
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def compute_route_entropy(mask):
    """
    Compute entropy of a route mask.
    High entropy = spread across many blocks (less selective).
    Low entropy = concentrated on few blocks (more selective).
    For a binary mask, entropy measures how far from uniform the selection is.
    """
    p = mask / (mask.sum() + 1e-8)
    p = p[p > 0]
    return -np.sum(p * np.log2(p + 1e-12))


def compute_block_frequency(route_masks, num_blocks):
    """
    Compute how often each block is selected across all tasks.
    Returns per-layer frequency vectors.
    """
    task_ids = list(route_masks.keys())
    if not task_ids:
        return {}

    num_layers = len(route_masks[task_ids[0]])
    freq = {}
    for l in range(num_layers):
        block_counts = np.zeros(num_blocks)
        for tid in task_ids:
            mask = np.array(route_masks[tid][l])
            block_counts += mask
        freq[l] = block_counts / len(task_ids)
    return freq


def compute_overlap_matrix(route_masks, task_ids):
    """
    Compute pairwise Jaccard overlap matrix for all task pairs.
    Returns a dict of matrices, one per layer.
    """
    num_layers = len(route_masks[task_ids[0]])
    n = len(task_ids)

    matrices = {}
    for l in range(num_layers):
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                m1 = np.array(route_masks[task_ids[i]][l])
                m2 = np.array(route_masks[task_ids[j]][l])
                intersection = (m1 * m2).sum()
                union = (m1 + m2 - m1 * m2).sum()
                mat[i, j] = intersection / max(union, 1e-8)
        matrices[l] = mat
    return matrices


def compute_route_statistics(route_masks, task_families=None):
    """
    Compute comprehensive route statistics.
    """
    task_ids = list(route_masks.keys())
    if not task_ids:
        return {}

    num_layers = len(route_masks[task_ids[0]])
    num_blocks = len(route_masks[task_ids[0]][0])

    stats = {
        "num_tasks": len(task_ids),
        "num_layers": num_layers,
        "num_blocks": num_blocks,
    }

    # Per-task entropy
    task_entropies = {}
    for tid in task_ids:
        entropies = []
        for l in range(num_layers):
            mask = np.array(route_masks[tid][l])
            entropies.append(compute_route_entropy(mask))
        task_entropies[tid] = {
            "mean": float(np.mean(entropies)),
            "std": float(np.std(entropies)),
            "per_layer": [float(e) for e in entropies],
        }
    stats["entropy"] = task_entropies

    # Block frequency
    freq = compute_block_frequency(route_masks, num_blocks)
    layer_freq_stats = {}
    for l, f in freq.items():
        layer_freq_stats[l] = {
            "mean": float(np.mean(f)),
            "std": float(np.std(f)),
            "min": float(np.min(f)),
            "max": float(np.max(f)),
            "most_used_blocks": [int(x) for x in np.argsort(f)[-5:][::-1]],
            "least_used_blocks": [int(x) for x in np.argsort(f)[:5]],
        }
    stats["block_frequency"] = layer_freq_stats

    # Per-task active fraction
    active_fractions = {}
    for tid in task_ids:
        fracs = []
        for l in range(num_layers):
            mask = np.array(route_masks[tid][l])
            fracs.append(float(mask.sum() / num_blocks))
        active_fractions[tid] = float(np.mean(fracs))
    stats["active_fractions"] = active_fractions

    # Overlap analysis
    if task_families:
        overlap_matrices = compute_overlap_matrix(route_masks, task_ids)

        # Average overlap matrix across layers
        avg_matrix = np.mean([overlap_matrices[l] for l in overlap_matrices], axis=0)

        within_family = []
        between_family = []
        for i, t1 in enumerate(task_ids):
            for j, t2 in enumerate(task_ids):
                if i >= j:
                    continue
                overlap = avg_matrix[i, j]
                if task_families.get(t1) == task_families.get(t2):
                    within_family.append(overlap)
                else:
                    between_family.append(overlap)

        stats["overlap"] = {
            "within_family_mean": float(np.mean(within_family)) if within_family else 0,
            "within_family_std": float(np.std(within_family)) if within_family else 0,
            "between_family_mean": float(np.mean(between_family)) if between_family else 0,
            "between_family_std": float(np.std(between_family)) if between_family else 0,
            "ratio": float(np.mean(within_family) / max(np.mean(between_family), 1e-8)) if within_family else 0,
        }

        # Per-family average entropy
        family_entropies = defaultdict(list)
        for tid in task_ids:
            fam = task_families.get(tid, "unknown")
            family_entropies[fam].append(task_entropies[tid]["mean"])
        stats["family_entropy"] = {
            fam: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for fam, vals in family_entropies.items()
        }

        # Per-layer within vs between overlap
        per_layer_overlap = {}
        for l in overlap_matrices:
            w, b = [], []
            for i, t1 in enumerate(task_ids):
                for j, t2 in enumerate(task_ids):
                    if i >= j:
                        continue
                    if task_families.get(t1) == task_families.get(t2):
                        w.append(overlap_matrices[l][i, j])
                    else:
                        b.append(overlap_matrices[l][i, j])
            per_layer_overlap[l] = {
                "within": float(np.mean(w)) if w else 0,
                "between": float(np.mean(b)) if b else 0,
            }
        stats["per_layer_overlap"] = per_layer_overlap

    return stats


def print_route_report(stats):
    """Pretty-print route analysis results."""
    print(f"\n{'='*60}")
    print(f"Route Analysis Report")
    print(f"{'='*60}")
    print(f"Tasks: {stats['num_tasks']}, Layers: {stats['num_layers']}, Blocks: {stats['num_blocks']}")

    # Entropy
    print(f"\n--- Route Entropy (higher = less selective) ---")
    for tid, ent in sorted(stats["entropy"].items(), key=lambda x: x[1]["mean"]):
        print(f"  {tid:20s}: {ent['mean']:.3f} +/- {ent['std']:.3f}")

    # Block frequency
    print(f"\n--- Block Frequency (per layer) ---")
    for l in sorted(stats["block_frequency"].keys()):
        bf = stats["block_frequency"][l]
        print(f"  Layer {l:2d}: mean={bf['mean']:.3f} std={bf['std']:.3f} "
              f"min={bf['min']:.3f} max={bf['max']:.3f} "
              f"most_used={bf['most_used_blocks'][:3]}")

    # Overlap
    if "overlap" in stats:
        ov = stats["overlap"]
        print(f"\n--- Overlap Analysis ---")
        print(f"  Within-family:  {ov['within_family_mean']:.3f} +/- {ov['within_family_std']:.3f}")
        print(f"  Between-family: {ov['between_family_mean']:.3f} +/- {ov['between_family_std']:.3f}")
        print(f"  Ratio: {ov['ratio']:.2f}x")

    # Family entropy
    if "family_entropy" in stats:
        print(f"\n--- Per-Family Entropy ---")
        for fam, ent in sorted(stats["family_entropy"].items()):
            print(f"  {fam:25s}: {ent['mean']:.3f} +/- {ent['std']:.3f}")

    # Per-layer overlap
    if "per_layer_overlap" in stats:
        print(f"\n--- Per-Layer Within vs Between Overlap ---")
        for l in sorted(stats["per_layer_overlap"].keys()):
            plo = stats["per_layer_overlap"][l]
            ratio = plo["within"] / max(plo["between"], 1e-8)
            print(f"  Layer {l:2d}: within={plo['within']:.3f} "
                  f"between={plo['between']:.3f} ratio={ratio:.2f}x")


def analyze_from_log(log_file, output_file=None):
    """Extract and analyze routes from a training log file."""
    print(f"Analyzing routes from {log_file}...")

    # Try to parse route data from JSON results
    with open(log_file) as f:
        content = f.read()

    # Look for RESULTS JSON
    if "RESULTS JSON" in content:
        json_start = content.index("RESULTS JSON") + len("RESULTS JSON") + 4
        json_end = content.index("END RESULTS")
        json_str = content[json_start:json_end].strip()
        try:
            results = json.loads(json_str)
            if "route_analysis" in results:
                print("Found route analysis in results:")
                ra = results["route_analysis"]
                for k, v in ra.items():
                    print(f"  {k}: {v}")
        except json.JSONDecodeError:
            print("Could not parse JSON from log")

    # Look for per-layer overlap data
    within_between = []
    for line in content.split('\n'):
        if 'within=' in line and 'between=' in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith('within='):
                    w = float(p.split('=')[1])
                elif p.startswith('between='):
                    b = float(p.split('=')[1])
            within_between.append((w, b))

    if within_between:
        print(f"\nPer-layer overlap from log ({len(within_between)} layers):")
        for i, (w, b) in enumerate(within_between):
            print(f"  Layer {i:2d}: within={w:.3f} between={b:.3f} ratio={w/max(b,1e-8):.2f}x")

    # Look for selected blocks data
    block_data = {}
    for line in content.split('\n'):
        if 'blocks [' in line and ':' in line:
            parts = line.strip().split(':')
            if len(parts) >= 2:
                tid = parts[0].strip()
                blocks_str = parts[1].strip()
                block_data[tid] = blocks_str

    if block_data:
        print(f"\nSelected blocks per task (from log):")
        for tid, blocks in block_data.items():
            print(f"  {tid}: {blocks[:60]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Route analysis")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Training log file to analyze")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory with result JSON files")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        analyze_from_log(args.log_file, args.output_file)
    else:
        # Analyze all available log files
        results_dir = args.results_dir
        log_files = [f for f in os.listdir(results_dir) if f.startswith("logs_")]
        for lf in sorted(log_files):
            print(f"\n{'='*60}")
            analyze_from_log(os.path.join(results_dir, lf))


if __name__ == "__main__":
    main()
