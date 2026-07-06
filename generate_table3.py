"""
Generate Table 3: Primary quality-efficiency table.

Collects results from all method eval JSON files and formats
into the paper's Table 3 format.

Columns: Method | Trainable M | Static M | Active FFN | Macro score |
         Worst 10% | Mean I_t | Tokens/s | Peak GB | T_{0.95}

Usage:
  python generate_table3.py --results_dir outputs/
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from models.baselines import get_method_summary


TABLE3_COLUMNS = [
    "Method",
    "Trainable M",
    "Static M",
    "Active FFN",
    "Macro score",
    "Worst 10%",
    "Mean I_t",
    "Tokens/s",
    "Peak GB",
    "T_0.95 h",
]


def load_results(results_dir: str) -> dict:
    """Load all eval result JSON files from a directory."""
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith(".json") and "eval" in fname:
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            method = data.get("mode", fname.replace(".json", ""))
            results[method] = data
    return results


def format_row(method_name: str, result: dict) -> dict:
    """Format a single result into a Table 3 row."""
    scores = result.get("scores", {})
    params = scores.get("params", {})
    efficiency = scores.get("efficiency", {})

    return {
        "Method": method_name,
        "Trainable M": f"{params.get('trainable_M', 0):.1f}",
        "Static M": "-",
        "Active FFN": result.get("active_ffn", "100%"),
        "Macro score": f"{scores.get('macro_avg', 0):.1f}",
        "Worst 10%": "RESULT TO FILL",
        "Mean I_t": "RESULT TO FILL",
        "Tokens/s": f"{efficiency.get('tokens_per_sec', 0):.0f}" if efficiency.get('tokens_per_sec', 0) > 0 else "RESULT TO FILL",
        "Peak GB": f"{efficiency.get('peak_gb', 0):.1f}" if efficiency.get('peak_gb', 0) > 0 else "RESULT TO FILL",
        "T_0.95 h": "RESULT TO FILL",
    }


def generate_table(results_dir: str = "outputs"):
    """Generate the full Table 3."""
    method_descriptions = get_method_summary()

    # Table 3 row order (from paper)
    row_order = [
        ("Frozen base", "frozen_base"),
        ("Full FFN fine-tuning", "full_ffn_finetune"),
        ("Dense multi-task LoRA", "dense_multitask_lora"),
        ("Task-specific LoRA", "task_specific_lora"),
        ("Task-family LoRA", "task_family_lora"),
        ("Direct Block-LoRA (50%)", "direct_block_lora"),
        ("Random-route mapped (50%)", "random_route_mapped"),
        ("TaskMap (25%)", "taskmap_25"),
        ("TaskMap (50%)", "taskmap_50"),
        ("TaskMap (75%)", "taskmap_75"),
    ]

    # Try to load existing results
    results = {}
    if os.path.isdir(results_dir):
        results = load_results(results_dir)

    # Print table
    print("\n" + "=" * 120)
    print("TABLE 3: Primary quality-efficiency table")
    print("=" * 120)
    header = " | ".join(f"{col:>15}" for col in TABLE3_COLUMNS)
    print(header)
    print("-" * 120)

    for display_name, method_key in row_order:
        if method_key in results:
            row = format_row(display_name, results[method_key])
        else:
            row = {col: "RESULT TO FILL" for col in TABLE3_COLUMNS}
            row["Method"] = display_name
            row["Active FFN"] = {
                "frozen_base": "100%",
                "full_ffn_finetune": "100%",
                "dense_multitask_lora": "100%",
                "task_specific_lora": "100%",
                "task_family_lora": "100%",
                "direct_block_lora": "50%",
                "random_route_mapped": "50%",
                "taskmap_25": "25%",
                "taskmap_50": "50%",
                "taskmap_75": "75%",
            }.get(method_key, "-")

        values = [f"{row.get(col, '-'):>15}" for col in TABLE3_COLUMNS]
        print(" | ".join(values))

    print("=" * 120)
    print("\nNote: 'RESULT TO FILL' entries require running the corresponding experiments.")
    print("Run each method's training script, then evaluate with run_eval.py.")

    # Also generate LaTeX
    print("\n% LaTeX version:")
    print("\\begin{tabular}{l" + "r" * (len(TABLE3_COLUMNS) - 1) + "}")
    print("\\toprule")
    print(" & ".join(TABLE3_COLUMNS) + " \\\\")
    print("\\midrule")
    for display_name, method_key in row_order:
        cells = [display_name]
        for col in TABLE3_COLUMNS[1:]:
            cells.append("\\textcolor{red}{RESULT TO FILL}")
        print(" & ".join(cells) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs")
    args = parser.parse_args()
    generate_table(args.results_dir)
