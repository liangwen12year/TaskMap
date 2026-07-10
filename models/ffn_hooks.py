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
        Replace MLP output with task-conditioned sparse FFN.

        This hook runs AFTER the original MLP forward. We recompute
        the output using only selected blocks + residuals.
        """
        if not self.active or self.route_info is None:
            return output

        # Get the input to the MLP (hidden states)
        hidden_states = args[0]  # (batch, seq_len, d)

        # Get the MLP weights
        gate_proj = module.gate_proj  # (d_ff, d)
        up_proj = module.up_proj      # (d_ff, d)
        down_proj = module.down_proj  # (d, d_ff)
        act_fn = module.act_fn if hasattr(module, 'act_fn') else F.silu

        selected = self.route_info['selected']
        # Keep coefficients in the graph — gradient flows:
        # task codes -> mapper -> coefficients -> residuals -> FFN output -> loss
        c_u = self.route_info['c_u']  # (G, r)
        c_g = self.route_info['c_g']  # (G, r)
        c_d = self.route_info['c_d']  # (G, r)
        b = self.block_size

        # Compute sparse FFN: only selected blocks
        gate_out = torch.zeros(
            *hidden_states.shape[:-1], len(selected) * b,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        up_out = torch.zeros_like(gate_out)

        for idx, g in enumerate(selected):
            start = g * b
            end = start + b

            # Gate projection for block g: h @ W^g[start:end, :].T = h @ (d, b)
            gate_weight_slice = gate_proj.weight[start:end, :]  # (b, d)
            gate_block = hidden_states @ gate_weight_slice.t()  # (batch, seq, b)

            # Add residual: dW^g has shape (d, b), so h @ dW^g -> (batch, seq, b)
            if self.residual_bases is not None:
                dW_g = self.residual_bases.compute_residual(
                    self.layer_idx, g, 'g', c_g[g].to(hidden_states.device)
                ).to(hidden_states.dtype)  # (d, b) for gate/up projections
                gate_block = gate_block + hidden_states @ dW_g

            gate_out[..., idx * b:(idx + 1) * b] = gate_block

            # Up projection for block g
            up_weight_slice = up_proj.weight[start:end, :]  # (b, d)
            up_block = hidden_states @ up_weight_slice.t()  # (batch, seq, b)

            # Add residual: dW^u has shape (d, b)
            if self.residual_bases is not None:
                dW_u = self.residual_bases.compute_residual(
                    self.layer_idx, g, 'u', c_u[g].to(hidden_states.device)
                ).to(hidden_states.dtype)  # (d, b)
                up_block = up_block + hidden_states @ dW_u

            up_out[..., idx * b:(idx + 1) * b] = up_block

        # SwiGLU activation
        intermediate = act_fn(gate_out) * up_out  # (batch, seq, k*b)

        # Down projection: only selected blocks of W^d
        result = torch.zeros(
            *hidden_states.shape, device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        for idx, g in enumerate(selected):
            start = g * b
            end = start + b

            down_weight_slice = down_proj.weight[:, start:end]  # (d, b)
            inter_block = intermediate[..., idx * b:(idx + 1) * b]  # (batch, seq, b)
            block_result = inter_block @ down_weight_slice.t()  # (batch, seq, d)

            # Add residual: dW^d has shape (b, d), so inter @ dW^d -> (batch, seq, d)
            if self.residual_bases is not None:
                dW_d = self.residual_bases.compute_residual(
                    self.layer_idx, g, 'd', c_d[g].to(hidden_states.device)
                ).to(hidden_states.dtype)  # (b, d) for down projection
                block_result = block_result + inter_block @ dW_d

            result = result + block_result

        return result

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
