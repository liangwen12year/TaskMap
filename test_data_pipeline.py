"""
Quick test: download 2 small datasets, format them, and run the sampler.
Verifies Step 1 of the pipeline end-to-end.

Usage: python test_data_pipeline.py
"""

import sys
sys.path.insert(0, ".")

from data.config import KNOWN_TASKS, TASK_FAMILIES, FAMILY_TO_TASKS, FAMILY_PAIRS
from data.format import format_dataset, FORMAT_FNS
from data.sampler import build_dataloader
from datasets import load_dataset
from collections import Counter


def test_config():
    print("=" * 60)
    print("TEST 1: Config integrity")
    print("=" * 60)
    assert len(KNOWN_TASKS) == 12, f"Expected 12 known tasks, got {len(KNOWN_TASKS)}"
    assert len(TASK_FAMILIES) == 6, f"Expected 6 families, got {len(TASK_FAMILIES)}"

    for tid, meta in KNOWN_TASKS.items():
        assert meta["family"] in TASK_FAMILIES, f"{tid} has unknown family {meta['family']}"
        assert len(meta["descriptions"]) == 3, f"{tid} needs 3 descriptions, has {len(meta['descriptions'])}"
        assert tid in FORMAT_FNS, f"{tid} missing from FORMAT_FNS"

    print(f"  12 tasks across {len(TASK_FAMILIES)} families")
    for fam, tasks in FAMILY_TO_TASKS.items():
        print(f"    {fam}: {tasks}")
    print(f"  {len(FAMILY_PAIRS)} within-family task pairs for topology loss")
    print("  PASSED\n")


def test_format_small():
    print("=" * 60)
    print("TEST 2: Format a small dataset (SST-2 validation)")
    print("=" * 60)
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    formatted = format_dataset("sst2", ds, split="validation", description_idx=0)
    print(f"  Formatted {len(formatted)} examples")
    print(f"  Sample:")
    ex = formatted[0]
    print(f"    task_id: {ex['task_id']}")
    print(f"    family: {ex['family']}")
    print(f"    description: {ex['description']}")
    print(f"    full_text:\n      {ex['full_text'][:300]}...")
    assert len(formatted) > 0
    assert all(k in formatted[0] for k in ["task_id", "family", "description", "input", "response", "full_text"])
    print("  PASSED\n")
    return formatted


def test_description_paraphrases():
    print("=" * 60)
    print("TEST 3: Description paraphrases produce different prompts")
    print("=" * 60)
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    texts = set()
    for idx in range(3):
        formatted = format_dataset("sst2", ds, split="validation", description_idx=idx)
        texts.add(formatted[0]["full_text"][:100])
        print(f"  Paraphrase {idx}: '{formatted[0]['description']}'")
    assert len(texts) == 3, "Paraphrases should produce different prompts"
    print("  PASSED\n")


def test_sampler():
    print("=" * 60)
    print("TEST 4: Task-homogeneous sampler")
    print("=" * 60)
    fake_data = {
        "sst2": [{"text": f"ex{i}"} for i in range(1000)],
        "gsm8k": [{"text": f"math{i}"} for i in range(200)],
        "xsum": [{"text": f"doc{i}"} for i in range(3000)],
    }
    task_counts = Counter()
    total_steps = 300
    for task_id, batch in build_dataloader(fake_data, microbatch_size=4, total_steps=total_steps):
        task_counts[task_id] += 1
        assert len(batch) == 4, f"Microbatch should have 4 examples, got {len(batch)}"

    print(f"  Distribution over {total_steps} steps (sqrt-proportional):")
    for tid, cnt in task_counts.most_common():
        size = len(fake_data[tid])
        print(f"    {tid} (n={size}): {cnt} batches ({cnt/total_steps:.1%})")

    assert task_counts["xsum"] > task_counts["gsm8k"], \
        "Larger dataset should be sampled more (sqrt-proportional)"
    print("  PASSED\n")


if __name__ == "__main__":
    test_config()
    test_format_small()
    test_description_paraphrases()
    test_sampler()
    print("=" * 60)
    print("ALL STEP 1 TESTS PASSED")
    print("=" * 60)
