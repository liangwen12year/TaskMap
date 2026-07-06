"""
Structured top-k routing (Section 3.4).

During warmup: soft gates with Gumbel noise, temperature annealed 1.0 -> 0.1
After warmup: straight-through exact top-k

The route is binary: m_{t,l} in {0,1}^G with exactly k = rho*G active blocks.
The route is task-static within a microbatch (computed once per task-layer per step).
"""

import torch
import torch.nn.functional as F


def gumbel_soft_topk(logits: torch.Tensor, k: int, temperature: float = 1.0,
                      hard: bool = False):
    """
    Soft top-k selection with Gumbel noise for differentiable routing.

    Args:
        logits: route logits (G,)
        k: number of blocks to select
        temperature: Gumbel temperature (anneal from 1.0 to 0.1)
        hard: if True, use straight-through estimator

    Returns:
        mask: soft or hard binary mask (G,)
    """
    if temperature > 0 and not hard:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        perturbed = (logits + gumbel_noise) / temperature
    else:
        perturbed = logits

    soft_mask = torch.sigmoid(perturbed)

    if hard:
        _, topk_indices = torch.topk(logits, k)
        hard_mask = torch.zeros_like(logits)
        hard_mask[topk_indices] = 1.0
        mask = hard_mask - soft_mask.detach() + soft_mask
    else:
        mask = soft_mask

    return mask


def hard_topk(logits: torch.Tensor, k: int):
    """
    Exact top-k selection with straight-through gradient.

    m_{t,l} = ST(1{q_{t,l} in TopK(q_{t,l}, k)})

    Args:
        logits: route logits (G,)
        k: number of blocks to select

    Returns:
        mask: binary mask (G,) with exactly k ones
    """
    _, topk_indices = torch.topk(logits, k)
    hard_mask = torch.zeros_like(logits)
    hard_mask[topk_indices] = 1.0

    soft_approx = torch.sigmoid(logits)
    mask = hard_mask - soft_approx.detach() + soft_approx

    return mask


class TopKRouter(torch.nn.Module):
    """
    Manages the routing schedule: Gumbel warmup then hard top-k.

    Schedule (Section 3.4, Table 2):
    - 3% dense warmup (all blocks active)
    - Gumbel temperature anneals from 1.0 to 0.1 via cosine decay
    - After warmup, use hard top-k with straight-through gradients
    """

    def __init__(self, num_blocks: int, active_fraction: float = 0.5,
                 warmup_fraction: float = 0.03, total_steps: int = 12000,
                 temp_start: float = 1.0, temp_end: float = 0.1):
        super().__init__()
        self.num_blocks = num_blocks
        self.k = max(1, int(active_fraction * num_blocks))
        self.warmup_steps = int(warmup_fraction * total_steps)
        self.total_steps = total_steps
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.current_step = 0

    def get_temperature(self):
        """Cosine decay from temp_start to temp_end over total steps."""
        if self.current_step <= self.warmup_steps:
            return self.temp_start
        progress = (self.current_step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        import math
        temp = self.temp_end + 0.5 * (self.temp_start - self.temp_end) * (
            1 + math.cos(math.pi * progress))
        return temp

    def is_warmup(self):
        return self.current_step < self.warmup_steps

    def route(self, logits: torch.Tensor):
        """
        Compute the route mask from logits.

        During warmup: all-ones mask (dense computation)
        During Gumbel phase: soft gates with noise
        After convergence: hard top-k

        Args:
            logits: route logits (G,) from mapper

        Returns:
            mask: (G,) tensor, values in [0, 1]
            selected: list of selected block indices
        """
        if self.is_warmup():
            mask = torch.ones_like(logits)
            selected = list(range(self.num_blocks))
        else:
            temp = self.get_temperature()
            use_hard = temp <= self.temp_end + 0.01
            if use_hard:
                mask = hard_topk(logits, self.k)
            else:
                mask = gumbel_soft_topk(logits, self.k, temp, hard=False)
            selected = torch.topk(logits, self.k).indices.tolist()

        return mask, selected

    def step(self):
        """Advance the schedule by one optimizer step."""
        self.current_step += 1
