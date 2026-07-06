"""
TaskMap loss functions (Section 3.6, Equations 9-15).

Total loss:
  L = L_task + lambda_bud * L_budget + lambda_topo * L_topology
    + lambda_bal * L_balance + lambda_stab * L_stability
    + lambda_sm * L_smooth + lambda_align * L_alignment

Reference hyperparameters (Table 2):
  lambda_bud=0.05, lambda_topo=0.01, lambda_bal=0.01
  lambda_stab=1e-3, lambda_sm=1e-3, lambda_align=1e-4
"""

import torch
import torch.nn.functional as F
import math


def budget_loss(soft_gates_per_layer: list, target_fraction: float):
    """
    L_budget (Eq. 10): enforce that soft gates average to target fraction rho.

    L = (1/L) * sum_l ( (1/G) * sum_g m_bar_{t,l,g} - rho )^2

    Only applied during Gumbel warmup when gates are soft.

    Args:
        soft_gates_per_layer: list of L tensors, each (G,) soft gate values
        target_fraction: rho (e.g., 0.5)
    """
    L = len(soft_gates_per_layer)
    if L == 0:
        return torch.tensor(0.0)

    loss = torch.tensor(0.0, device=soft_gates_per_layer[0].device)
    for gates in soft_gates_per_layer:
        mean_activation = gates.mean()
        loss = loss + (mean_activation - target_fraction) ** 2
    return loss / L


def topology_loss(route_masks: dict, family_pairs: list, task_families: dict,
                  pi_near: float = 0.8, pi_far: float = 0.2):
    """
    L_topology (Eq. 11): encourage same-family tasks to share routes.

    For each task pair (t, t'):
      a_{tt'} = 1 if same family, 0 otherwise
      target = pi_near if a=1, pi_far if a=0
      o^l_{tt'} = soft Jaccard overlap of route probabilities at layer l
      L = (1/|P|L) * sum_{(t,t')} sum_l (o^l_{tt'} - target)^2

    Args:
        route_masks: {task_id: list of L (G,) mask tensors}
        family_pairs: list of (task_id_1, task_id_2) same-family pairs
        task_families: {task_id: family_name}
        pi_near: target overlap for same-family pairs
        pi_far: target overlap for different-family pairs
    """
    if not route_masks or len(route_masks) < 2:
        return torch.tensor(0.0)

    task_ids = list(route_masks.keys())
    num_layers = len(route_masks[task_ids[0]])
    device = route_masks[task_ids[0]][0].device

    same_family_set = set()
    for t1, t2 in family_pairs:
        same_family_set.add((t1, t2))
        same_family_set.add((t2, t1))

    loss = torch.tensor(0.0, device=device)
    count = 0

    for i, t1 in enumerate(task_ids):
        for t2 in task_ids[i + 1:]:
            is_near = (t1, t2) in same_family_set
            target = pi_near if is_near else pi_far

            for l in range(num_layers):
                m1 = route_masks[t1][l]
                m2 = route_masks[t2][l]
                intersection = (m1 * m2).sum()
                union = (m1 + m2 - m1 * m2).sum().clamp(min=1e-8)
                jaccard = intersection / union
                loss = loss + (jaccard - target) ** 2
                count += 1

    return loss / max(count, 1)


def balance_loss(route_masks: dict, num_blocks: int):
    """
    L_balance (Eq. 12): prevent all tasks from selecting the same blocks.

    L = (1/L) * sum_l KL(p_bar_l || Uniform(G))

    where p_bar_{l,g} = (1/|T|) * sum_t m_{t,l,g}

    Applied only after initial warmup.

    Args:
        route_masks: {task_id: list of L (G,) mask tensors}
        num_blocks: G
    """
    if not route_masks:
        return torch.tensor(0.0)

    task_ids = list(route_masks.keys())
    num_layers = len(route_masks[task_ids[0]])
    device = route_masks[task_ids[0]][0].device

    uniform = torch.ones(num_blocks, device=device) / num_blocks
    loss = torch.tensor(0.0, device=device)

    for l in range(num_layers):
        avg_usage = torch.zeros(num_blocks, device=device)
        for tid in task_ids:
            avg_usage = avg_usage + route_masks[tid][l]
        avg_usage = avg_usage / len(task_ids)
        avg_usage = avg_usage.clamp(min=1e-8)
        avg_usage = avg_usage / avg_usage.sum()
        loss = loss + F.kl_div(avg_usage.log(), uniform, reduction='sum')

    return loss / num_layers


def stability_loss(mapper_fn, z: torch.Tensor, sigma: float = 0.01,
                   epsilon: float = 1e-8):
    """
    L_stability (Eq. 13): mapper output should be stable under small input perturbation.

    L = E_delta [ ||G(z + delta) - G(z)||^2 / (||delta||^2 + eps) ]

    Args:
        mapper_fn: callable that takes z and returns (q, c_u, c_g, c_d)
        z: task code (d_z,)
        sigma: noise std
        epsilon: numerical stability
    """
    delta = torch.randn_like(z) * sigma
    with torch.no_grad():
        out_clean = mapper_fn(z)
        out_noisy = mapper_fn(z + delta)

    out_clean_cat = torch.cat([o.flatten() for o in out_clean])
    out_noisy_cat = torch.cat([o.flatten() for o in out_noisy])

    diff_norm_sq = (out_noisy_cat - out_clean_cat).pow(2).sum()
    delta_norm_sq = delta.pow(2).sum() + epsilon

    return diff_norm_sq / delta_norm_sq


def smoothness_loss(mapper_fn, z: torch.Tensor):
    """
    L_smooth (Eq. 14): Jacobian norm via Hutchinson estimator.

    L = E_v [ ||J_{G_l}(z) v||^2 ]

    where v ~ N(0, I) and J_{G_l}(z) v is computed via a single JVP.

    Args:
        mapper_fn: callable that takes z and returns (q, c_u, c_g, c_d)
        z: task code (d_z,), must have requires_grad=True
    """
    v = torch.randn_like(z)

    z_var = z.detach().requires_grad_(True)
    out = mapper_fn(z_var)
    out_cat = torch.cat([o.flatten() for o in out])

    jvp = torch.autograd.grad(
        out_cat, z_var, grad_outputs=torch.ones_like(out_cat),
        create_graph=False, retain_graph=False,
    )[0]

    return (jvp * v).pow(2).sum()


def alignment_loss(z: torch.Tensor, mapper_output: tuple,
                    R: torch.Tensor):
    """
    L_alignment (Eq. 15): code should align with mapper output direction.

    L = 1 - cos(z / ||z||, R @ o / ||R @ o||)

    where R is a fixed random projection matrix.

    Args:
        z: task code (d_z,)
        mapper_output: (q, c_u, c_g, c_d) from mapper
        R: fixed random projection (d_z, output_dim)
    """
    o = torch.cat([t.flatten() for t in mapper_output])
    projected = R @ o

    z_norm = F.normalize(z.unsqueeze(0), dim=-1)
    p_norm = F.normalize(projected.unsqueeze(0), dim=-1)

    cos_sim = (z_norm * p_norm).sum()
    return 1.0 - cos_sim


class TaskMapLossComputer:
    """
    Computes all TaskMap losses for a training step.
    """

    def __init__(self, config, family_pairs: list, task_families: dict,
                 lambda_bud: float = 0.05, lambda_topo: float = 0.01,
                 lambda_bal: float = 0.01, lambda_stab: float = 1e-3,
                 lambda_sm: float = 1e-3, lambda_align: float = 1e-4):
        self.config = config
        self.family_pairs = family_pairs
        self.task_families = task_families
        self.lambda_bud = lambda_bud
        self.lambda_topo = lambda_topo
        self.lambda_bal = lambda_bal
        self.lambda_stab = lambda_stab
        self.lambda_sm = lambda_sm
        self.lambda_align = lambda_align

        output_dim = config.num_blocks + 3 * config.num_blocks * config.rank
        self.R = torch.randn(config.code_dim, output_dim) * (1.0 / math.sqrt(output_dim))

    def compute(self, task_loss: torch.Tensor, taskmap_model, current_task: str,
                all_route_masks: dict, is_warmup: bool):
        """
        Compute total loss.

        Args:
            task_loss: L_task (cross-entropy)
            taskmap_model: the TaskMapModel
            current_task: task_id of current microbatch
            all_route_masks: {task_id: list of L (G,) masks} for all seen tasks
            is_warmup: whether in dense warmup phase

        Returns:
            total_loss, loss_dict
        """
        device = task_loss.device
        self.R = self.R.to(device)

        losses = {"task": task_loss}
        total = task_loss

        # Budget loss (only during soft-gate phase)
        if is_warmup and current_task in all_route_masks:
            masks = all_route_masks[current_task]
            l_bud = budget_loss(masks, self.config.active_fraction)
            losses["budget"] = l_bud
            total = total + self.lambda_bud * l_bud

        # Topology loss (after warmup, needs multiple tasks)
        if not is_warmup and len(all_route_masks) > 1:
            l_topo = topology_loss(all_route_masks, self.family_pairs,
                                   self.task_families)
            losses["topology"] = l_topo
            total = total + self.lambda_topo * l_topo

        # Balance loss (after warmup)
        if not is_warmup and len(all_route_masks) > 1:
            l_bal = balance_loss(all_route_masks, self.config.num_blocks)
            losses["balance"] = l_bal
            total = total + self.lambda_bal * l_bal

        # Stability loss (sample one layer)
        if self.lambda_stab > 0 and current_task in all_route_masks:
            import random
            l_idx = random.randint(0, self.config.num_layers - 1)
            z = taskmap_model.task_code.get_code(current_task, l_idx, device)
            mapper_fn = lambda z_in: taskmap_model.mapper_bank(l_idx, z_in)
            l_stab = stability_loss(mapper_fn, z)
            losses["stability"] = l_stab
            total = total + self.lambda_stab * l_stab

        # Alignment loss (sample one layer)
        if self.lambda_align > 0 and current_task in all_route_masks:
            import random
            l_idx = random.randint(0, self.config.num_layers - 1)
            z = taskmap_model.task_code.get_code(current_task, l_idx, device)
            mapper_out = taskmap_model.mapper_bank(l_idx, z)
            l_align = alignment_loss(z, mapper_out, self.R)
            losses["alignment"] = l_align
            total = total + self.lambda_align * l_align

        losses["total"] = total
        return total, losses
