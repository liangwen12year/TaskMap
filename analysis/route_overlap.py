"""
Route analysis: compute Jaccard overlap between task routes (Section 5.4).

Compares within-family vs between-family route overlap to verify that
related tasks share more blocks than unrelated tasks.
"""

import torch
import numpy as np
from collections import defaultdict


def jaccard_overlap(mask1, mask2):
    """Compute Jaccard overlap between two binary masks."""
    intersection = (mask1 * mask2).sum().item()
    union = ((mask1 + mask2) > 0).float().sum().item()
    return intersection / max(union, 1e-8)


def compute_route_overlaps(taskmap_model, task_ids, task_families, device="cpu"):
    """
    Compute layerwise Jaccard overlap for all task pairs.

    Returns:
        overlaps: dict of {(t1, t2): [overlap_per_layer]}
        within_family: average overlap for same-family pairs
        between_family: average overlap for different-family pairs
    """
    # Compute routes for all tasks
    routes = {}
    for tid in task_ids:
        taskmap_model.clear_route_cache()
        route = taskmap_model.compute_route(tid, device)
        routes[tid] = [r['mask'].detach() for r in route]

    num_layers = len(routes[task_ids[0]])

    # Compute pairwise overlaps
    overlaps = {}
    within_scores = []
    between_scores = []

    for i, t1 in enumerate(task_ids):
        for t2 in task_ids[i + 1:]:
            layer_overlaps = []
            for l in range(num_layers):
                j = jaccard_overlap(routes[t1][l], routes[t2][l])
                layer_overlaps.append(j)
            overlaps[(t1, t2)] = layer_overlaps
            avg = np.mean(layer_overlaps)

            if task_families[t1] == task_families[t2]:
                within_scores.append(avg)
            else:
                between_scores.append(avg)

    within_avg = np.mean(within_scores) if within_scores else 0
    between_avg = np.mean(between_scores) if between_scores else 0

    return overlaps, within_avg, between_avg


def print_route_report(overlaps, within_avg, between_avg, task_families):
    """Print a formatted route overlap report."""
    print("=" * 60)
    print("Route Overlap Analysis")
    print("=" * 60)
    print(f"\nWithin-family average overlap:  {within_avg:.3f}")
    print(f"Between-family average overlap: {between_avg:.3f}")
    print(f"Ratio (within/between):         {within_avg / max(between_avg, 1e-8):.2f}x")

    print("\nPairwise overlaps (averaged across layers):")
    for (t1, t2), layer_overlaps in sorted(overlaps.items()):
        avg = np.mean(layer_overlaps)
        same_fam = task_families.get(t1) == task_families.get(t2)
        marker = " [SAME FAMILY]" if same_fam else ""
        print(f"  {t1:15s} <-> {t2:15s}: {avg:.3f}{marker}")
