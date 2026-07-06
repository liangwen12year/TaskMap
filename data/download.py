"""
Download all datasets for TaskMap-12 known tasks and 6 cold-start tasks.
Uses HuggingFace datasets library. Run once to cache locally.
"""

from datasets import load_dataset
from data.config import KNOWN_TASKS, COLD_START_TASKS


def download_task(task_id: str, meta: dict) -> dict:
    """Download a single task's dataset and return split dict."""
    print(f"Downloading {task_id} ({meta['dataset']})...")
    kwargs = {}
    if meta["subset"]:
        kwargs["name"] = meta["subset"]
    try:
        ds = load_dataset(meta["dataset"], **kwargs, trust_remote_code=True)
        print(f"  Splits: {list(ds.keys())}")
        return ds
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def download_all(include_cold_start: bool = False):
    """Download all known tasks and optionally cold-start tasks."""
    results = {}

    print("=" * 60)
    print("Downloading KNOWN TASKS (12 tasks)")
    print("=" * 60)
    for task_id, meta in KNOWN_TASKS.items():
        ds = download_task(task_id, meta)
        if ds is not None:
            results[task_id] = ds

    if include_cold_start:
        print("\n" + "=" * 60)
        print("Downloading COLD-START TASKS (6 tasks)")
        print("=" * 60)
        for task_id, meta in COLD_START_TASKS.items():
            ds = download_task(task_id, meta)
            if ds is not None:
                results[task_id] = ds

    print(f"\nSuccessfully downloaded {len(results)} datasets.")
    return results


if __name__ == "__main__":
    download_all(include_cold_start=True)
