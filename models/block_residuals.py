"""
Block-local low-rank residuals (Section 3.5).

For each selected block g in layer l, the weight residual is:
  delta_W^u_{t,l,g} = A^u_{l,g} @ diag(c^u_{t,l,g}) @ B^u_{l,g}

where:
- A^u_{l,g} in R^{d x r}: fixed orthogonal random basis (left)
- B^u_{l,g} in R^{r x b}: fixed orthogonal random basis (right)
- c^u_{t,l,g} in R^r: coefficients from mapper (task-dependent)

Same structure for gate (W^g) and down (W^d) projections.
Bases are frozen; only coefficients (from mapper) are used.
"""

import torch
import torch.nn as nn


def create_orthogonal_basis(rows: int, cols: int, rank: int):
    """
    Create fixed orthogonal random bases A (rows x rank) and B (rank x cols).
    """
    A = torch.empty(rows, rank)
    B = torch.empty(rank, cols)
    nn.init.orthogonal_(A)
    nn.init.orthogonal_(B)
    return A, B


class BlockResidualBases(nn.Module):
    """
    Stores fixed orthogonal bases for all layers, blocks, and projections.
    These are frozen and never updated.

    For a model with:
    - L layers, G blocks per layer, 3 projections (up, gate, down)
    - Each projection has bases A (left) and B (right)
    """

    def __init__(self, num_layers: int, num_blocks: int, rank: int,
                 model_dim: int, block_size: int):
        """
        Args:
            num_layers: L
            num_blocks: G = d_ff / block_size
            rank: r (low-rank dimension)
            model_dim: d (hidden dimension of the Transformer)
            block_size: b (contiguous neuron block size)
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.rank = rank
        self.model_dim = model_dim
        self.block_size = block_size

        self.bases = nn.ParameterDict()

        for l in range(num_layers):
            for g in range(num_blocks):
                # Up projection: W^u in R^{d_ff x d}, block slice is R^{b x d}
                # A^u: (d, r), B^u: (r, b)
                A_u, B_u = create_orthogonal_basis(model_dim, block_size, rank)
                self.bases[f"l{l}_g{g}_u_A"] = nn.Parameter(A_u, requires_grad=False)
                self.bases[f"l{l}_g{g}_u_B"] = nn.Parameter(B_u, requires_grad=False)

                # Gate projection: same shape as up
                A_g, B_g = create_orthogonal_basis(model_dim, block_size, rank)
                self.bases[f"l{l}_g{g}_g_A"] = nn.Parameter(A_g, requires_grad=False)
                self.bases[f"l{l}_g{g}_g_B"] = nn.Parameter(B_g, requires_grad=False)

                # Down projection: W^d in R^{d x d_ff}, block slice is R^{d x b}
                # A^d: (block_size, r), B^d: (r, d)
                # Note: down proj maps from d_ff to d, so block is along input dim
                A_d, B_d = create_orthogonal_basis(block_size, model_dim, rank)
                self.bases[f"l{l}_g{g}_d_A"] = nn.Parameter(A_d, requires_grad=False)
                self.bases[f"l{l}_g{g}_d_B"] = nn.Parameter(B_d, requires_grad=False)

        for p in self.parameters():
            p.requires_grad = False

    def compute_residual(self, layer_idx: int, block_idx: int,
                          proj_type: str, coefficients: torch.Tensor):
        """
        Compute delta_W = A @ diag(c) @ B for one block and projection.

        Args:
            layer_idx: layer index l
            block_idx: block index g
            proj_type: 'u' (up), 'g' (gate), or 'd' (down)
            coefficients: (r,) tensor from mapper

        Returns:
            delta_W: weight residual matrix
        """
        A = self.bases[f"l{layer_idx}_g{block_idx}_{proj_type}_A"]
        B = self.bases[f"l{layer_idx}_g{block_idx}_{proj_type}_B"]
        delta_W = A @ torch.diag(coefficients) @ B
        return delta_W

    def memory_bytes(self):
        """Report total memory for all bases."""
        total = sum(p.numel() * p.element_size() for p in self.parameters())
        return total
