"""
FFN hooks for TaskMap (Section 3.5, Equations 6-8).

Injects task-conditioned block selection and low-rank residuals into
the backbone's MLP layers via PyTorch forward hooks. This is the key
integration point: the hook replaces the standard dense FFN with the
TaskMap sparse FFN that only activates selected blocks.

TaskFFN_{t,l}(h) = sum_{g in S_{t,l}} [
    phi(h @ [W^g_{:,I_g} + dW^g]) * (h @ [W^u_{:,I_g} + dW^u])
] @ [W^d_{I_g,:} + dW^d]

where dW = A @ diag(c) @ B are the low-rank residuals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskMapFFNHook:
    """
    Hooks into a single MLP layer to apply task-conditioned sparse FFN.

    When active, replaces the MLP forward with block-sparse computation
    using the route and coefficients from the TaskMap model.
    """

    def __init__(self, layer_idx: int, block_size: int):
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.active = False
        self.route_info = None  # set per forward pass
        self.residual_bases = None  # reference to BlockResidualBases
        self.handle = None

    def set_route(self, route_info: dict, residual_bases):
        """
        Set the route for this layer's next forward pass.

        Args:
            route_info: dict with 'mask', 'selected', 'c_u', 'c_g', 'c_d'
            residual_bases: BlockResidualBases module
        """
        self.route_info = route_info
        self.residual_bases = residual_bases
        self.active = True

    def deactivate(self):
        self.active = False
        self.route_info = None

    def hook_fn(self, module, args, output):
        """
        ADDITIVE approach: keep dense MLP output, add task-specific residuals.

        output = dense_FFN(h) + residual_contribution(h)

        The residual contribution computes low-rank modifications only on
        selected blocks, using the coefficients from the mapper/DBL:
          residual = h @ dW_combined
        where dW_combined is built from the selected blocks' A @ diag(c) @ B.

        This preserves the base model's full computation while adding
        task-specific adaptations on top.
        """
        if not self.active or self.route_info is None:
            return output

        hidden_states = args[0]  # (batch, seq_len, d)
        gate_proj = module.gate_proj
        up_proj = module.up_proj
        down_proj = module.down_proj
        act_fn = module.act_fn if hasattr(module, 'act_fn') else F.silu

        selected = self.route_info['selected']
        c_u = self.route_info['c_u']
        c_g = self.route_info['c_g']
        c_d = self.route_info['c_d']
        b = self.block_size
        k = len(selected)
        dtype = hidden_states.dtype
        device = hidden_states.device

        if self.residual_bases is None or k == 0:
            return output

        # Build index for selected blocks
        indices = []
        for g in selected:
            indices.extend(range(g * b, (g + 1) * b))
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        kb = k * b

        # Compute residual contribution for gate and up projections
        # dW_g, dW_u: (d, kb) — residuals for selected blocks only
        dW_g_parts = []
        dW_u_parts = []
        for g in selected:
            dW_g_parts.append(
                self.residual_bases.compute_residual(
                    self.layer_idx, g, 'g', c_g[g].to(device)
                ).to(dtype)
            )
            dW_u_parts.append(
                self.residual_bases.compute_residual(
                    self.layer_idx, g, 'u', c_u[g].to(device)
                ).to(dtype)
            )
        dW_g_full = torch.cat(dW_g_parts, dim=1)  # (d, kb)
        dW_u_full = torch.cat(dW_u_parts, dim=1)  # (d, kb)

        # Residual gate/up: h @ dW -> (batch, seq, kb)
        res_gate = hidden_states @ dW_g_full
        res_up = hidden_states @ dW_u_full

        # Get the base gate/up outputs for selected blocks only
        gate_w_sel = gate_proj.weight[idx_tensor, :]  # (kb, d)
        up_w_sel = up_proj.weight[idx_tensor, :]      # (kb, d)
        base_gate = hidden_states @ gate_w_sel.t()    # (batch, seq, kb)
        base_up = hidden_states @ up_w_sel.t()        # (batch, seq, kb)

        # Modified = base + residual for selected blocks
        mod_gate = base_gate + res_gate
        mod_up = base_up + res_up

        # Compute difference in activation: modified - original for selected blocks
        # original_act = act_fn(base_gate) * base_up
        # modified_act = act_fn(mod_gate) * mod_up
        # delta_act = modified_act - original_act
        original_act = act_fn(base_gate) * base_up
        modified_act = act_fn(mod_gate) * mod_up
        delta_act = modified_act - original_act  # (batch, seq, kb)

        # Down projection of the delta through base weights + residual
        down_w_sel = down_proj.weight[:, idx_tensor]  # (d, kb)
        delta_output = delta_act @ down_w_sel.t()     # (batch, seq, d)

        # Add down-projection residuals
        dW_d_parts = []
        for g in selected:
            dW_d_parts.append(
                self.residual_bases.compute_residual(
                    self.layer_idx, g, 'd', c_d[g].to(device)
                ).to(dtype)
            )
        dW_d_full = torch.cat(dW_d_parts, dim=0)  # (kb, d)
        delta_output = delta_output + modified_act @ dW_d_full

        # Additive: dense output + task-specific delta
        return output + delta_output

    def register(self, mlp_module):
        """Register the hook on an MLP module."""
        self.handle = mlp_module.register_forward_hook(self.hook_fn, with_kwargs=False)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TaskMapHookManager:
    """
    Manages FFN hooks across all layers of the backbone.
    """

    def __init__(self, backbone_model, taskmap_model, block_size: int = 128):
        self.backbone = backbone_model
        self.taskmap = taskmap_model
        self.block_size = block_size
        self.hooks = []
        self._install_hooks()

    def _get_mlp_layers(self):
        """Find all MLP modules in the backbone."""
        mlp_layers = []
        # Qwen2 / Llama style
        if hasattr(self.backbone, 'model'):
            model = self.backbone.model
        elif hasattr(self.backbone, 'base_model'):
            model = self.backbone.base_model.model
        else:
            model = self.backbone

        if hasattr(model, 'layers'):
            for layer in model.layers:
                mlp_layers.append(layer.mlp)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                mlp_layers.append(layer.mlp)

        return mlp_layers

    def _install_hooks(self):
        """Install hooks on all MLP layers."""
        mlp_layers = self._get_mlp_layers()
        for idx, mlp in enumerate(mlp_layers):
            hook = TaskMapFFNHook(idx, self.block_size)
            hook.register(mlp)
            self.hooks.append(hook)
        print(f"Installed TaskMap hooks on {len(self.hooks)} MLP layers")

    def activate_for_task(self, task_id: str, device: str = "cuda"):
        """
        Compute routes and activate hooks for a specific task.
        Call this once per task per optimizer step.
        """
        routes = self.taskmap.compute_route(task_id, device)
        for layer_idx, hook in enumerate(self.hooks):
            if layer_idx < len(routes):
                hook.set_route(routes[layer_idx], self.taskmap.residual_bases)

    def deactivate(self):
        """Deactivate all hooks (fall back to dense FFN)."""
        for hook in self.hooks:
            hook.deactivate()

    def remove_all(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
