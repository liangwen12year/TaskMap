"""
Task code: description prior + learned residual (Section 3.2).

For each task t and layer l:
  z_{t,l} = P_l @ e_t + r_{t,l}

where:
- e_t: frozen description embedding (mean-pooled backbone input embeddings, layer-normed)
- P_l: small learned projector per layer (d_z x d_e)
- r_{t,l}: learned residual code per task per layer, init ~N(0, 1e-4)

Cold-start: r_{t*,l} = 0 (description prior only).
Few-shot: optimize r_{t,l} only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskCodeModule(nn.Module):
    """
    Produces per-layer task codes z_{t,l} from a task description
    and optional learned residuals.
    """

    def __init__(self, num_layers: int, embed_dim: int, code_dim: int,
                 num_tasks: int, task_descriptions: dict = None,
                 shared_projector: bool = False, global_code: bool = False):
        """
        Args:
            num_layers: number of Transformer layers L
            embed_dim: dimension of backbone input embeddings d_e
            code_dim: dimension of task code d_z
            num_tasks: number of known training tasks T
            task_descriptions: {task_id: str} for computing description priors
            shared_projector: if True, share one projector across all layers (Mapping Networks param reduction)
            global_code: if True, use one residual code per task (not per task-layer)
        """
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.code_dim = code_dim
        self.num_tasks = num_tasks
        self.shared_projector = shared_projector
        self.global_code = global_code

        if shared_projector:
            self._shared_proj = nn.Linear(embed_dim, code_dim, bias=False)
            self.projectors = nn.ModuleList([self._shared_proj] * num_layers)
        else:
            self.projectors = nn.ModuleList([
                nn.Linear(embed_dim, code_dim, bias=False)
                for _ in range(num_layers)
            ])

        self.residuals = nn.ParameterDict()
        self.task_id_to_idx = {}

        self.description_cache = {}

    def _sanitize_key(self, name):
        """Replace characters that PyTorch doesn't allow in parameter names."""
        return name.replace(".", "_dot_")

    def register_tasks(self, task_ids: list):
        """Register known tasks and initialize their residual codes."""
        for idx, tid in enumerate(task_ids):
            self.task_id_to_idx[tid] = idx
            safe_tid = self._sanitize_key(tid)
            if self.global_code:
                key = f"{safe_tid}_global"
                self.residuals[key] = nn.Parameter(
                    torch.randn(self.code_dim) * 1e-4
                )
            else:
                for l in range(self.num_layers):
                    key = f"{safe_tid}_layer{l}"
                    self.residuals[key] = nn.Parameter(
                        torch.randn(self.code_dim) * 1e-4
                    )

    def cache_description_embedding(self, task_id: str, embedding: torch.Tensor):
        """
        Cache the frozen description prior e_t for a task.
        embedding: (d_e,) tensor from mean-pooled backbone input embeddings.
        """
        self.description_cache[task_id] = embedding.detach()

    @torch.no_grad()
    def compute_description_embedding(self, backbone_model, tokenizer,
                                       description: str, device: str = "cpu"):
        """
        Compute e_t = LayerNorm(mean(embed(description_tokens))).
        Uses frozen backbone input embeddings only (no forward pass).
        """
        tokens = tokenizer(description, return_tensors="pt", truncation=True,
                           max_length=128).to(device)
        if hasattr(backbone_model, 'model'):
            embed_layer = backbone_model.model.embed_tokens
        elif hasattr(backbone_model, 'transformer'):
            embed_layer = backbone_model.transformer.wte
        else:
            embed_layer = backbone_model.get_input_embeddings()

        token_embeds = embed_layer(tokens["input_ids"])  # (1, seq_len, d_e)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        mean_embed = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1)  # (1, d_e)
        e_t = F.layer_norm(mean_embed.squeeze(0), [mean_embed.size(-1)])
        return e_t

    def get_code(self, task_id: str, layer_idx: int, device: str = "cpu"):
        """
        Compute z_{t,l} = P_l @ e_t + r_{t,l}.
        For cold-start (unknown task_id), r = 0.
        """
        if task_id not in self.description_cache:
            raise ValueError(f"Description embedding not cached for '{task_id}'. "
                             f"Call cache_description_embedding() first.")

        e_t = self.description_cache[task_id].to(device)
        projected = self.projectors[layer_idx](e_t)  # (d_z,)

        safe_tid = self._sanitize_key(task_id)
        residual_key = f"{safe_tid}_global" if self.global_code else f"{safe_tid}_layer{layer_idx}"
        if residual_key in self.residuals:
            r = self.residuals[residual_key].to(device)
            z = projected + r
        else:
            z = projected

        return z

    def get_all_layer_codes(self, task_id: str, device: str = "cpu"):
        """Get codes for all layers at once. Returns list of (d_z,) tensors."""
        return [self.get_code(task_id, l, device) for l in range(self.num_layers)]

    def trainable_parameters(self):
        """Return only the parameters that should be optimized."""
        params = []
        for proj in self.projectors:
            params.extend(proj.parameters())
        for name, param in self.residuals.items():
            params.append(param)
        return params

    def num_trainable(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.projectors.parameters())
        total += sum(p.numel() for p in self.residuals.values())
        return total
