"""
Additional baselines for Table 3 (Section 4.5).

Methods 4-9 beyond frozen base, full FFN fine-tuning, and dense multi-task LoRA:
4. Task-specific LoRA — one adapter per known task (interference-minimizing)
5. Task-family LoRA — one adapter per 6 families
6. Adapter tuning — bottleneck adapters at matched parameter budget
7. Direct Block-LoRA — block routes + coefficients optimized directly (no mapper)
8. Random-route mapped — mapper generates coefficients, routes are random
9. No-description TaskMap — learned task IDs replace description embeddings
"""

import torch
import torch.nn as nn
import copy
from models.backbone import add_lora
from models.taskmap_model import TaskMapModel, TaskMapConfig


class TaskSpecificLoRA:
    """
    Method 4: One LoRA adapter per known task.
    At eval, load the task's adapter. Upper bound on per-task quality.
    Storage scales linearly with number of tasks.
    """

    def __init__(self, base_model, task_ids: list, rank: int = 16):
        self.base_model = base_model
        self.task_ids = task_ids
        self.rank = rank
        self.adapters = {}

    def get_model_for_task(self, task_id: str):
        """Return a model with task-specific LoRA."""
        if task_id not in self.adapters:
            model_copy = copy.deepcopy(self.base_model)
            model_with_lora = add_lora(model_copy, rank=self.rank)
            self.adapters[task_id] = model_with_lora
        return self.adapters[task_id]

    def trainable_params_per_task(self):
        if not self.adapters:
            dummy = add_lora(copy.deepcopy(self.base_model), rank=self.rank)
            return sum(p.numel() for p in dummy.parameters() if p.requires_grad)
        tid = list(self.adapters.keys())[0]
        return sum(p.numel() for p in self.adapters[tid].parameters() if p.requires_grad)

    def total_stored_params(self):
        return self.trainable_params_per_task() * len(self.task_ids)


class TaskFamilyLoRA:
    """
    Method 5: One LoRA adapter per task family (6 families).
    Tests whether coarse manual grouping is sufficient.
    """

    def __init__(self, base_model, family_to_tasks: dict, rank: int = 16):
        self.base_model = base_model
        self.family_to_tasks = family_to_tasks
        self.rank = rank
        self.family_adapters = {}

        # Reverse mapping
        self.task_to_family = {}
        for fam, tasks in family_to_tasks.items():
            for t in tasks:
                self.task_to_family[t] = fam

    def get_model_for_task(self, task_id: str):
        family = self.task_to_family[task_id]
        if family not in self.family_adapters:
            model_copy = copy.deepcopy(self.base_model)
            model_with_lora = add_lora(model_copy, rank=self.rank)
            self.family_adapters[family] = model_with_lora
        return self.family_adapters[family]


class DirectBlockLoRA(nn.Module):
    """
    Method 7: Identical block routes and low-rank basis shapes,
    but route logits and coefficients are directly optimized per task
    (no mapper). Isolates the Mapping Networks contribution.
    """

    def __init__(self, config: TaskMapConfig, task_ids: list):
        super().__init__()
        self.config = config
        self.task_ids = task_ids

        # Direct per-task route logits (no mapper)
        self.route_logits = nn.ParameterDict()
        for tid in task_ids:
            for l in range(config.num_layers):
                self.route_logits[f"{tid}_l{l}"] = nn.Parameter(
                    torch.randn(config.num_blocks) * 0.01
                )

        # Direct per-task coefficients (no mapper)
        self.coefficients = nn.ParameterDict()
        for tid in task_ids:
            for l in range(config.num_layers):
                for proj in ['u', 'g', 'd']:
                    self.coefficients[f"{tid}_l{l}_{proj}"] = nn.Parameter(
                        torch.randn(config.num_blocks, config.rank) * 0.01
                    )

    def get_route(self, task_id: str, layer_idx: int):
        logits = self.route_logits[f"{task_id}_l{layer_idx}"]
        k = max(1, int(self.config.active_fraction * self.config.num_blocks))
        _, topk = torch.topk(logits, k)
        mask = torch.zeros_like(logits)
        mask[topk] = 1.0
        return mask, topk.tolist(), logits

    def get_coefficients(self, task_id: str, layer_idx: int):
        c_u = self.coefficients[f"{task_id}_l{layer_idx}_u"]
        c_g = self.coefficients[f"{task_id}_l{layer_idx}_g"]
        c_d = self.coefficients[f"{task_id}_l{layer_idx}_d"]
        return c_u, c_g, c_d

    def num_trainable(self):
        return sum(p.numel() for p in self.parameters())


class RandomRouteTaskMap(nn.Module):
    """
    Method 8: Mapper generates residual coefficients, but each task receives
    a fixed random route (not learned). Tests whether learned routing matters.
    """

    def __init__(self, config: TaskMapConfig, task_ids: list, seed: int = 42):
        super().__init__()
        self.config = config
        k = max(1, int(config.active_fraction * config.num_blocks))

        # Fixed random routes per task per layer
        gen = torch.Generator().manual_seed(seed)
        self.random_routes = {}
        for tid in task_ids:
            routes = []
            for l in range(config.num_layers):
                perm = torch.randperm(config.num_blocks, generator=gen)[:k]
                mask = torch.zeros(config.num_blocks)
                mask[perm] = 1.0
                routes.append({
                    'mask': mask,
                    'selected': perm.tolist(),
                })
            self.random_routes[tid] = routes

    def get_route(self, task_id: str, layer_idx: int):
        route = self.random_routes[task_id][layer_idx]
        return route['mask'], route['selected']


class NoDescriptionTaskMap(nn.Module):
    """
    Method 9: Learned task IDs replace description embeddings.
    Tests the description prior and cold-start claim.
    """

    def __init__(self, config: TaskMapConfig, task_ids: list):
        super().__init__()
        self.config = config
        self.task_embeddings = nn.Embedding(len(task_ids), config.code_dim)
        self.task_id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
        nn.init.normal_(self.task_embeddings.weight, std=0.02)

    def get_code(self, task_id: str, layer_idx: int):
        idx = self.task_id_to_idx[task_id]
        idx_tensor = torch.tensor(idx, device=self.task_embeddings.weight.device)
        return self.task_embeddings(idx_tensor)


def get_method_summary():
    """Return a description of all methods for Table 3."""
    return {
        "frozen_base": "No adaptation, frozen backbone only",
        "full_ffn_finetune": "Full FFN fine-tuning (Tier A only, upper-capacity reference)",
        "dense_multitask_lora": "One shared FFN LoRA adapter, ranks {8, 16, 32}",
        "task_specific_lora": "One FFN LoRA adapter per known task",
        "task_family_lora": "One adapter per 6 task families",
        "adapter_tuning": "Bottleneck adapters at matched parameter budget",
        "direct_block_lora": "Block routes + coefficients optimized directly (no mapper)",
        "random_route_mapped": "Mapper generates coefficients, routes are random",
        "no_description_taskmap": "Learned task IDs replace description embeddings",
        "taskmap_25": "TaskMap at 25% active fraction",
        "taskmap_50": "TaskMap at 50% active fraction (reference)",
        "taskmap_75": "TaskMap at 75% active fraction",
    }
