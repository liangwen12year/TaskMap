"""
Frozen modulated mapper G_l (Section 3.3).

For each layer l, maps task code z_{t,l} to:
  (q_{t,l}, c^u_{t,l}, c^g_{t,l}, c^d_{t,l}) = G_l(z_{t,l})

where:
- q: route logits (G-dim, one per FFN block)
- c^u, c^g, c^d: low-rank residual coefficients (G x r each)

The mapper is:
- Two orthogonally-initialized linear stages with GELU
- Latent-code-dependent modulation (Mapping Networks principle):
  the fixed map is changed by a bounded affine function of z at each stage
- Frozen during training: only codes z are optimized
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def orthogonal_init_(tensor):
    """Initialize a matrix with orthogonal initialization."""
    if tensor.ndim < 2:
        nn.init.normal_(tensor, std=0.02)
        return tensor
    return nn.init.orthogonal_(tensor)


class ModulatedLinear(nn.Module):
    """
    Linear layer with input-dependent affine modulation.
    out = W @ (x * (1 + gamma(z))) + b + beta(z)
    where gamma and beta are small linear projections of z.
    """

    def __init__(self, in_features: int, out_features: int, code_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gamma_proj = nn.Linear(code_dim, in_features, bias=False)
        self.beta_proj = nn.Linear(code_dim, out_features, bias=False)

        orthogonal_init_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, x, z):
        """
        Args:
            x: input features (..., in_features)
            z: task code (d_z,)
        """
        gamma = self.gamma_proj(z)  # (in_features,)
        beta = self.beta_proj(z)    # (out_features,)
        modulated_x = x * (1.0 + gamma)
        return self.linear(modulated_x) + beta


class FrozenMapper(nn.Module):
    """
    Per-layer frozen mapper that produces route logits and
    low-rank residual coefficients from a task code.

    Output dimension: G + 3*G*r = G(1 + 3r)
    For b=128, d_ff=8960: G = 8960/128 = 70
    With r=8: output = 70*(1+24) = 1750
    """

    def __init__(self, code_dim: int, num_blocks: int, rank: int,
                 hidden_dim: int = 512):
        """
        Args:
            code_dim: dimension of input task code d_z
            num_blocks: number of FFN blocks G = d_ff / b
            rank: low-rank residual dimension r
            hidden_dim: mapper hidden width
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.rank = rank

        self.stage1 = ModulatedLinear(code_dim, hidden_dim, code_dim)
        self.stage2 = ModulatedLinear(hidden_dim, hidden_dim, code_dim)

        output_dim = num_blocks + 3 * num_blocks * rank
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        orthogonal_init_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z):
        """
        Args:
            z: task code (d_z,)
        Returns:
            q: route logits (G,)
            c_u: up-projection coefficients (G, r)
            c_g: gate-projection coefficients (G, r)
            c_d: down-projection coefficients (G, r)
        """
        h = F.gelu(self.stage1(z, z))
        h = F.gelu(self.stage2(h, z))
        out = self.output_proj(h)  # (G + 3*G*r,)

        G = self.num_blocks
        r = self.rank

        q = out[:G]                                    # (G,)
        c_u = out[G:G + G * r].view(G, r)             # (G, r)
        c_g = out[G + G * r:G + 2 * G * r].view(G, r) # (G, r)
        c_d = out[G + 2 * G * r:].view(G, r)          # (G, r)

        return q, c_u, c_g, c_d

    def freeze(self):
        """Freeze all mapper parameters (they don't receive gradients)."""
        for p in self.parameters():
            p.requires_grad = False


class MapperBank(nn.Module):
    """Collection of per-layer mappers. Can be frozen or trainable."""

    def __init__(self, num_layers: int, code_dim: int, num_blocks: int,
                 rank: int, hidden_dim: int = 512, frozen: bool = True):
        super().__init__()
        self.mappers = nn.ModuleList([
            FrozenMapper(code_dim, num_blocks, rank, hidden_dim)
            for _ in range(num_layers)
        ])
        if frozen:
            self.freeze()

    def freeze(self):
        for mapper in self.mappers:
            mapper.freeze()

    def forward(self, layer_idx: int, z):
        return self.mappers[layer_idx](z)

    def forward_all_layers(self, codes: list):
        """
        Args:
            codes: list of L task codes, each (d_z,)
        Returns:
            list of (q, c_u, c_g, c_d) tuples
        """
        return [self.mappers[l](codes[l]) for l in range(len(self.mappers))]
