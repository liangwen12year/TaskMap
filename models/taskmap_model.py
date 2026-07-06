"""
Full TaskMap model wrapper (Section 3, Equations 6-8).

Assembles:
1. Frozen LLM backbone
2. Task code module (description prior + residuals)
3. Mapper bank (frozen, modulated)
4. Top-k router
5. Block residual bases

The TaskFFN for selected blocks (Eq. 6-8):
  TaskFFN_{t,l}(h) = sum_{g in S_{t,l}} [
    phi(h @ [W^g_{l,:,I_g} + delta_W^g_{t,l,g}])
    * (h @ [W^u_{l,:,I_g} + delta_W^u_{t,l,g}])
  ] @ [W^d_{l,I_g,:} + delta_W^d_{t,l,g}]

Inactive blocks are skipped entirely (not computed then masked).
Dense-mask mode computes all blocks then masks for correctness verification.
"""

import torch
import torch.nn as nn

from models.task_code import TaskCodeModule
from models.mapper import MapperBank
from models.router import TopKRouter
from models.block_residuals import BlockResidualBases


class TaskMapConfig:
    """Configuration for TaskMap."""

    def __init__(
        self,
        num_layers: int = 28,
        model_dim: int = 1536,
        ffn_dim: int = 8960,
        block_size: int = 128,
        active_fraction: float = 0.50,
        code_dim: int = 32,
        rank: int = 8,
        mapper_hidden: int = 512,
        embed_dim: int = 1536,
        total_steps: int = 12000,
        warmup_fraction: float = 0.03,
    ):
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.block_size = block_size
        self.num_blocks = ffn_dim // block_size  # G
        self.active_fraction = active_fraction
        self.code_dim = code_dim
        self.rank = rank
        self.mapper_hidden = mapper_hidden
        self.embed_dim = embed_dim
        self.total_steps = total_steps
        self.warmup_fraction = warmup_fraction

    @classmethod
    def from_backbone(cls, backbone_name: str, **overrides):
        """Create config matching a specific backbone."""
        configs = {
            "Qwen/Qwen2.5-0.5B": dict(num_layers=24, model_dim=896, ffn_dim=4864, embed_dim=896),
            "Qwen/Qwen2.5-1.5B": dict(num_layers=28, model_dim=1536, ffn_dim=8960, embed_dim=1536),
            "Qwen/Qwen2.5-7B": dict(num_layers=28, model_dim=3584, ffn_dim=18944, embed_dim=3584),
            "meta-llama/Llama-3.2-1B": dict(num_layers=16, model_dim=2048, ffn_dim=8192, embed_dim=2048),
            "meta-llama/Llama-3.1-8B": dict(num_layers=32, model_dim=4096, ffn_dim=14336, embed_dim=4096),
        }
        base = configs.get(backbone_name, {})
        base.update(overrides)
        return cls(**base)


class TaskMapModel(nn.Module):
    """
    TaskMap: task-conditioned mapped sparse adaptation.

    This module wraps around a frozen LLM and provides the TaskMap
    components. It does NOT modify the LLM forward pass directly —
    instead, it provides methods to compute task-specific FFN modifications
    that should be applied during forward.
    """

    def __init__(self, config: TaskMapConfig, num_tasks: int):
        super().__init__()
        self.config = config

        self.task_code = TaskCodeModule(
            num_layers=config.num_layers,
            embed_dim=config.embed_dim,
            code_dim=config.code_dim,
            num_tasks=num_tasks,
        )

        self.mapper_bank = MapperBank(
            num_layers=config.num_layers,
            code_dim=config.code_dim,
            num_blocks=config.num_blocks,
            rank=config.rank,
            hidden_dim=config.mapper_hidden,
        )

        self.router = TopKRouter(
            num_blocks=config.num_blocks,
            active_fraction=config.active_fraction,
            warmup_fraction=config.warmup_fraction,
            total_steps=config.total_steps,
        )

        self.residual_bases = BlockResidualBases(
            num_layers=config.num_layers,
            num_blocks=config.num_blocks,
            rank=config.rank,
            model_dim=config.model_dim,
            block_size=config.block_size,
        )

        self._route_cache = {}

    def register_tasks(self, task_ids: list):
        """Register known tasks."""
        self.task_code.register_tasks(task_ids)

    def cache_description(self, task_id: str, embedding: torch.Tensor):
        """Cache a task's description embedding."""
        self.task_code.cache_description_embedding(task_id, embedding)

    def compute_route(self, task_id: str, device: str = "cpu"):
        """
        Compute the full route for a task: codes -> mapper -> top-k selection.
        Cached per task per optimizer step.

        Returns:
            routes: list of L dicts, each with:
                - 'mask': (G,) binary mask
                - 'selected': list of selected block indices
                - 'c_u': (G, r) up coefficients
                - 'c_g': (G, r) gate coefficients
                - 'c_d': (G, r) down coefficients
        """
        if task_id in self._route_cache:
            return self._route_cache[task_id]

        codes = self.task_code.get_all_layer_codes(task_id, device)
        routes = []

        for l in range(self.config.num_layers):
            q, c_u, c_g, c_d = self.mapper_bank(l, codes[l])
            mask, selected = self.router.route(q)

            routes.append({
                'mask': mask,
                'selected': selected,
                'c_u': c_u,
                'c_g': c_g,
                'c_d': c_d,
                'logits': q,
            })

        self._route_cache[task_id] = routes
        return routes

    def clear_route_cache(self):
        """Clear cached routes (call once per optimizer step)."""
        self._route_cache = {}

    def get_block_residual(self, layer_idx: int, block_idx: int,
                            proj_type: str, coefficients: torch.Tensor):
        """Compute delta_W for one block."""
        return self.residual_bases.compute_residual(
            layer_idx, block_idx, proj_type, coefficients)

    def step(self):
        """Advance router schedule and clear cache."""
        self.router.step()
        self.clear_route_cache()

    def trainable_parameters(self):
        """Only task codes (projectors + residuals) are trainable."""
        return self.task_code.trainable_parameters()

    def parameter_summary(self):
        """Report parameter counts by category."""
        trainable = self.task_code.num_trainable()
        mapper_params = sum(p.numel() for p in self.mapper_bank.parameters())
        bases_bytes = self.residual_bases.memory_bytes()
        return {
            "trainable (codes + projectors)": trainable,
            "frozen mapper": mapper_params,
            "frozen bases (bytes)": bases_bytes,
            "frozen bases (MB)": bases_bytes / 1e6,
        }
