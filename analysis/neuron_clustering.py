"""
Functional neuron clustering for FFN layers.

Instead of contiguous blocks, cluster neurons by activation similarity
so that functionally related neurons are grouped together. Then create
a permutation that makes these clusters contiguous, allowing the same
block-based routing code to operate on functional groups.

The permutation is applied to W^u, W^g (columns) and W^d (rows) of each
FFN layer. Model behavior is unchanged because the intermediate dimension
is internal to the FFN.

Usage:
  python -m analysis.neuron_clustering \
    --backbone Qwen/Qwen2.5-1.5B \
    --num_examples 500 \
    --output_file cluster_permutations.pt
"""

import os
import sys
import argparse
import torch
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.backbone import load_backbone
from data.config import KNOWN_TASKS
from data.download import download_task
from data.format import format_all_tasks
from train import tokenize_batch, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Neuron clustering")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--method", type=str, default="activation",
                        choices=["activation", "gradient", "weight"])
    parser.add_argument("--output_file", type=str, default="cluster_permutations.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_mlp_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return [(i, layer.mlp) for i, layer in enumerate(model.model.layers)]
    elif hasattr(model, 'layers'):
        return [(i, layer.mlp) for i, layer in enumerate(model.layers)]
    return []


def collect_activations(model, tokenizer, examples, device, max_seq=512):
    """Collect per-neuron activation magnitudes across examples."""
    mlp_layers = get_mlp_layers(model)
    num_layers = len(mlp_layers)

    # Storage: per-layer list of activation vectors
    activations = {i: [] for i in range(num_layers)}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output shape: [batch, seq, d_ff] for gate/up, or [batch, seq, d] for down
            # We want intermediate activations. For Qwen2 MLP:
            # gate_proj -> act_fn -> * up_proj -> down_proj
            # The input to down_proj is the intermediate activation
            pass
        return hook_fn

    # Instead of hooks, compute activations manually per layer
    model.eval()
    with torch.no_grad():
        for ex_idx, ex in enumerate(examples):
            if ex_idx >= len(examples):
                break
            inputs = tokenizer(ex["full_text"], return_tensors="pt",
                             truncation=True, max_length=max_seq).to(device)

            # Forward through the model, collecting hidden states
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of [1, seq, d]

            for layer_idx, (_, mlp) in enumerate(mlp_layers):
                h = hidden_states[layer_idx]  # [1, seq, d]
                # Compute gate and up projections
                if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj'):
                    gate = mlp.act_fn(mlp.gate_proj(h))  # [1, seq, d_ff]
                    up = mlp.up_proj(h)  # [1, seq, d_ff]
                    intermediate = gate * up  # [1, seq, d_ff]
                elif hasattr(mlp, 'w1') and hasattr(mlp, 'w2'):
                    gate = torch.nn.functional.silu(mlp.w1(h))
                    up = mlp.w3(h)
                    intermediate = gate * up
                else:
                    continue

                # Mean absolute activation per neuron across sequence
                neuron_act = intermediate.abs().mean(dim=(0, 1))  # [d_ff]
                activations[layer_idx].append(neuron_act.cpu())

            if (ex_idx + 1) % 50 == 0:
                print(f"  Processed {ex_idx + 1}/{len(examples)} examples")

    # Stack into matrices: [num_examples, d_ff] per layer
    act_matrices = {}
    for layer_idx in activations:
        if activations[layer_idx]:
            act_matrices[layer_idx] = torch.stack(activations[layer_idx])
    return act_matrices


def cluster_neurons(act_matrix, num_clusters, seed=42):
    """
    Cluster neurons by activation similarity using k-means.
    Returns cluster assignments and a permutation that groups clusters contiguously.
    """
    # Normalize each neuron's activation profile
    act_np = act_matrix.float().numpy().T  # [d_ff, num_examples]
    norms = np.linalg.norm(act_np, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    act_normalized = act_np / norms

    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(act_normalized)

    # Create permutation: sort neurons by cluster assignment
    # Within each cluster, sort by distance to cluster center (most representative first)
    permutation = []
    cluster_sizes = []
    for c in range(num_clusters):
        members = np.where(labels == c)[0]
        # Sort by distance to center
        center = kmeans.cluster_centers_[c]
        dists = np.linalg.norm(act_normalized[members] - center, axis=1)
        sorted_members = members[np.argsort(dists)]
        permutation.extend(sorted_members.tolist())
        cluster_sizes.append(len(members))

    return labels, np.array(permutation), cluster_sizes


def compute_cluster_quality(act_matrix, labels, num_clusters):
    """Compute intra-cluster vs inter-cluster similarity."""
    act_np = act_matrix.float().numpy().T  # [d_ff, num_examples]
    norms = np.linalg.norm(act_np, axis=1, keepdims=True)
    act_normalized = act_np / np.maximum(norms, 1e-8)

    intra_sims = []
    inter_sims = []

    for c in range(num_clusters):
        members = np.where(labels == c)[0]
        non_members = np.where(labels != c)[0]
        if len(members) < 2:
            continue

        # Intra-cluster: mean pairwise cosine similarity
        member_vecs = act_normalized[members]
        sim_matrix = member_vecs @ member_vecs.T
        n = len(members)
        intra_sim = (sim_matrix.sum() - n) / max(n * (n - 1), 1)
        intra_sims.append(intra_sim)

        # Inter-cluster: mean similarity with non-members (sample for speed)
        if len(non_members) > 100:
            sample_idx = np.random.choice(len(non_members), 100, replace=False)
            non_member_vecs = act_normalized[non_members[sample_idx]]
        else:
            non_member_vecs = act_normalized[non_members]
        inter_sim = (member_vecs @ non_member_vecs.T).mean()
        inter_sims.append(inter_sim)

    return np.mean(intra_sims), np.mean(inter_sims)


def apply_permutation_to_model(model, permutations):
    """
    Apply neuron permutations to FFN weight matrices.
    Permutes columns of W^u, W^g and rows of W^d.
    Model behavior is unchanged.
    """
    mlp_layers = get_mlp_layers(model)
    for layer_idx, (_, mlp) in enumerate(mlp_layers):
        if layer_idx not in permutations:
            continue
        perm = permutations[layer_idx]
        perm_tensor = torch.tensor(perm, dtype=torch.long)

        with torch.no_grad():
            if hasattr(mlp, 'gate_proj'):
                mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[perm_tensor]
                mlp.up_proj.weight.data = mlp.up_proj.weight.data[perm_tensor]
                mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, perm_tensor]
            elif hasattr(mlp, 'w1'):
                mlp.w1.weight.data = mlp.w1.weight.data[perm_tensor]
                mlp.w3.weight.data = mlp.w3.weight.data[perm_tensor]
                mlp.w2.weight.data = mlp.w2.weight.data[:, perm_tensor]

    print(f"Applied permutations to {len(permutations)} layers")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading backbone: {args.backbone}")
    model, tokenizer = load_backbone(args.backbone)
    model = model.to(device)
    model.eval()

    # Load data
    print("Loading data for activation collection...")
    datasets = {}
    for tid, meta in KNOWN_TASKS.items():
        ds = download_task(tid, meta)
        if ds is not None:
            datasets[tid] = ds
    all_data = format_all_tasks(datasets, split="train")

    # Sample examples evenly across tasks
    examples = []
    per_task = max(1, args.num_examples // len(all_data))
    for tid, task_examples in all_data.items():
        examples.extend(task_examples[:per_task])
    examples = examples[:args.num_examples]
    print(f"Collected {len(examples)} examples across {len(all_data)} tasks")

    # Collect activations
    print("\nCollecting neuron activations...")
    act_matrices = collect_activations(model, tokenizer, examples, device)

    # Determine number of clusters per layer
    mlp_layers = get_mlp_layers(model)
    d_ff = act_matrices[0].shape[1]
    num_clusters = d_ff // args.block_size
    print(f"\nd_ff = {d_ff}, block_size = {args.block_size}, num_clusters = {num_clusters}")

    # Cluster each layer
    print("\nClustering neurons per layer...")
    permutations = {}
    all_labels = {}
    for layer_idx in range(len(mlp_layers)):
        if layer_idx not in act_matrices:
            continue
        labels, perm, cluster_sizes = cluster_neurons(
            act_matrices[layer_idx], num_clusters, seed=args.seed
        )
        permutations[layer_idx] = perm
        all_labels[layer_idx] = labels

        intra, inter = compute_cluster_quality(
            act_matrices[layer_idx], labels, num_clusters
        )
        size_std = np.std(cluster_sizes)
        print(f"  Layer {layer_idx:2d}: intra={intra:.3f} inter={inter:.3f} "
              f"ratio={intra/max(inter, 1e-8):.2f}x "
              f"sizes: mean={np.mean(cluster_sizes):.0f} std={size_std:.0f} "
              f"min={min(cluster_sizes)} max={max(cluster_sizes)}")

    # Save permutations
    torch.save({
        "permutations": permutations,
        "labels": all_labels,
        "backbone": args.backbone,
        "block_size": args.block_size,
        "num_clusters": num_clusters,
        "num_examples": len(examples),
        "method": args.method,
    }, args.output_file)
    print(f"\nSaved cluster permutations to {args.output_file}")

    # Verify permutation preserves model behavior
    print("\nVerifying permutation correctness...")
    test_input = tokenizer("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        out_before = model(**test_input).logits[0, -1, :5].clone()

    apply_permutation_to_model(model, permutations)

    with torch.no_grad():
        out_after = model(**test_input).logits[0, -1, :5]

    diff = (out_before - out_after).abs().max().item()
    print(f"  Max logit difference after permutation: {diff:.6f}")
    if diff < 1e-3:
        print("  PASSED: Permutation preserves model behavior")
    else:
        print("  WARNING: Non-trivial difference detected")


if __name__ == "__main__":
    main()
