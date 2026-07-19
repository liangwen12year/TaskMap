"""
TaskMap training with functional neuron clusters instead of contiguous blocks.

Loads precomputed cluster permutations, applies them to the backbone's FFN
weights, then trains TaskMap normally. Since the permutation groups functionally
similar neurons contiguously, the block-based routing now operates on
functional modules rather than arbitrary slices.

Usage:
  python train_taskmap_clustered.py \
    --config configs/taskmap_reference.yaml \
    --cluster_file cluster_permutations.pt
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))

from analysis.neuron_clustering import apply_permutation_to_model
from train_taskmap import parse_args as taskmap_parse_args, load_config, train_taskmap
from train import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="TaskMap with clustered blocks")
    parser.add_argument("--config", type=str, default="configs/taskmap_reference.yaml")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active_fraction", type=float, default=None)
    parser.add_argument("--unfreeze_mapper", action="store_true")
    parser.add_argument("--mapping_loss", action="store_true")
    parser.add_argument("--shared_projector", action="store_true")
    parser.add_argument("--global_code", action="store_true")
    parser.add_argument("--code_dim", type=int, default=None)
    parser.add_argument("--microbatch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--cluster_file", type=str, required=True,
                        help="Path to cluster permutations from neuron_clustering.py")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load cluster permutations
    print(f"Loading cluster permutations from {args.cluster_file}")
    cluster_data = torch.load(args.cluster_file, map_location="cpu")
    permutations = cluster_data["permutations"]
    print(f"  Backbone: {cluster_data['backbone']}")
    print(f"  Block size: {cluster_data['block_size']}")
    print(f"  Num clusters: {cluster_data['num_clusters']}")
    print(f"  Method: {cluster_data['method']}")
    print(f"  Layers with permutations: {len(permutations)}")

    # Monkey-patch load_backbone to apply permutation after loading
    from models import backbone as backbone_module
    original_load_backbone = backbone_module.load_backbone

    def load_backbone_with_clustering(backbone_name):
        model, tokenizer = original_load_backbone(backbone_name)
        print("\n=== Applying neuron cluster permutations ===")
        apply_permutation_to_model(model, permutations)
        print("=== Permutations applied — contiguous blocks are now functional clusters ===\n")
        return model, tokenizer

    backbone_module.load_backbone = load_backbone_with_clustering

    # Set output dir to distinguish from non-clustered runs
    if args.output_dir is None:
        args.output_dir = "outputs/taskmap_clustered"

    # Run standard TaskMap training
    train_taskmap(args)


if __name__ == "__main__":
    main()
