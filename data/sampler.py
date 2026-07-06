"""
Task-homogeneous microbatch sampler for TaskMap training.

From the paper (Section 4.3-4.4):
- Each microbatch contains examples from exactly ONE task
- Sampling probability proportional to n_t^0.5 (capped training count)
- Smaller sources sampled with replacement up to 2x oversampling cap
- The route is task-static within a microbatch (computed once, shared)
"""

import math
import random
from typing import Dict, List
from data.config import OVERSAMPLING_CAP

try:
    from torch.utils.data import Sampler as _Sampler
except ImportError:
    _Sampler = object


class TaskHomogeneousSampler(_Sampler):
    """
    Yields batches where all examples come from the same task.
    Tasks are sampled proportionally to sqrt(n_t).
    """

    def __init__(
        self,
        task_data: Dict[str, List[dict]],
        microbatch_size: int = 4,
        total_steps: int = 12000,
        seed: int = 42,
    ):
        self.task_ids = list(task_data.keys())
        self.task_data = task_data
        self.microbatch_size = microbatch_size
        self.total_steps = total_steps
        self.rng = random.Random(seed)

        self.task_sizes = {tid: len(data) for tid, data in task_data.items()}
        raw_weights = {tid: math.sqrt(n) for tid, n in self.task_sizes.items()}
        total_w = sum(raw_weights.values())
        self.task_probs = {tid: w / total_w for tid, w in raw_weights.items()}

        self.task_indices = {}
        for tid, data in task_data.items():
            indices = list(range(len(data)))
            max_draws = min(len(data) * OVERSAMPLING_CAP, self.total_steps * microbatch_size)
            if len(indices) < max_draws:
                repeats = math.ceil(max_draws / len(indices))
                indices = (indices * repeats)[:max_draws]
            self.rng.shuffle(indices)
            self.task_indices[tid] = indices

        self.task_cursors = {tid: 0 for tid in self.task_ids}

    def _sample_task(self) -> str:
        r = self.rng.random()
        cumulative = 0.0
        for tid in self.task_ids:
            cumulative += self.task_probs[tid]
            if r <= cumulative:
                return tid
        return self.task_ids[-1]

    def _get_batch_indices(self, task_id: str) -> List[int]:
        indices = self.task_indices[task_id]
        cursor = self.task_cursors[task_id]
        batch = []
        for _ in range(self.microbatch_size):
            if cursor >= len(indices):
                self.rng.shuffle(indices)
                cursor = 0
            batch.append(indices[cursor])
            cursor += 1
        self.task_cursors[task_id] = cursor
        return batch

    def __iter__(self):
        for _ in range(self.total_steps):
            task_id = self._sample_task()
            batch_indices = self._get_batch_indices(task_id)
            yield task_id, batch_indices

    def __len__(self):
        return self.total_steps


def build_dataloader(task_data: Dict[str, List[dict]], microbatch_size: int = 4,
                     total_steps: int = 12000, seed: int = 42):
    """
    Returns an iterator that yields (task_id, list_of_examples) per step.
    Each step is a task-homogeneous microbatch.
    """
    sampler = TaskHomogeneousSampler(task_data, microbatch_size, total_steps, seed)
    for task_id, indices in sampler:
        examples = [task_data[task_id][i] for i in indices]
        yield task_id, examples


if __name__ == "__main__":
    fake_data = {
        "sst2": [{"text": f"example {i}"} for i in range(1000)],
        "gsm8k": [{"text": f"math {i}"} for i in range(500)],
        "xsum": [{"text": f"doc {i}"} for i in range(2000)],
    }
    from collections import Counter
    task_counts = Counter()
    for task_id, batch in build_dataloader(fake_data, microbatch_size=4, total_steps=100):
        task_counts[task_id] += 1
    print("Task sampling distribution over 100 steps:")
    for tid, cnt in task_counts.most_common():
        print(f"  {tid}: {cnt} batches ({cnt/100:.0%})")
