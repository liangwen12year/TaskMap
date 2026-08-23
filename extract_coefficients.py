"""Extract coefficients, codes, PCA, and nearest-task analysis from a saved checkpoint."""
import torch, json, numpy as np, sys
sys.path.insert(0, '.')
from models.backbone import load_backbone
from models.taskmap_model import TaskMapModel, TaskMapConfig
from data.config import KNOWN_TASKS, PAPER_FIGURE4_TASKS
from train import set_seed
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load('outputs/taskmap_no_topology/final/taskmap_state.pt', map_location='cpu')
cfg = ckpt['config']
task_ids = [t for t in PAPER_FIGURE4_TASKS if t in KNOWN_TASKS]

tm_config = TaskMapConfig.from_backbone(cfg.get('backbone', 'Qwen/Qwen2.5-1.5B'),
    block_size=cfg.get('block_size', 128), active_fraction=cfg.get('active_fraction', 0.75),
    code_dim=cfg.get('code_dim', 32), rank=cfg.get('rank', 8), mapper_hidden=cfg.get('mapper_hidden', 512))
taskmap = TaskMapModel(tm_config, num_tasks=len(task_ids), freeze_mapper=False)
taskmap.register_tasks(task_ids)
loaded_keys = taskmap.task_code.load_state_dict(ckpt['task_code_state'], strict=False)
if loaded_keys.missing_keys:
    missing_tasks = set()
    for k in loaded_keys.missing_keys:
        if 'residuals.' in k:
            parts = k.split('.')
            for p in parts:
                if '_layer' in p:
                    tid = p.rsplit('_layer', 1)[0]
                    missing_tasks.add(tid)
    if missing_tasks:
        raise RuntimeError(
            f"Checkpoint is missing trained residuals for tasks: {missing_tasks}. "
            f"Train with --figure4_tasks to include all 12 Figure 4 tasks."
        )
if 'mapper_state' in ckpt:
    taskmap.mapper_bank.load_state_dict(ckpt['mapper_state'], strict=False)
taskmap = taskmap.to(device)

backbone, tokenizer = load_backbone(cfg.get('backbone', 'Qwen/Qwen2.5-1.5B'))
backbone = backbone.to(device)
backbone.eval()

all_coefficients = {}
all_codes = {}
for tid in task_ids:
    desc = KNOWN_TASKS[tid]['descriptions'][0]
    embed = taskmap.task_code.compute_description_embedding(backbone, tokenizer, desc, device)
    taskmap.cache_description(tid, embed)
    layer_coeffs = []
    layer_codes = []
    for l in range(tm_config.num_layers):
        z = taskmap.task_code.get_code(tid, l, device)
        layer_codes.append(z.detach().cpu().numpy())
        out = taskmap.mapper_bank(l, z)
        q, cu, cg, cd = out
        coeffs = torch.cat([cu.flatten(), cg.flatten(), cd.flatten()]).detach().cpu().numpy()
        layer_coeffs.append(coeffs)
    all_coefficients[tid] = np.stack(layer_coeffs)
    all_codes[tid] = np.stack(layer_codes)
    print(f'  {tid}: coeff shape={all_coefficients[tid].shape}')

avg_coeffs = np.stack([v.mean(axis=0) for v in all_coefficients.values()])
pca = PCA(n_components=2)
coords = pca.fit_transform(avg_coeffs)
families = [KNOWN_TASKS[tid]['family'] for tid in task_ids]

print('\nPCA of average coefficients:')
for tid, (x, y), fam in zip(task_ids, coords, families):
    print(f'  {tid:15s} ({fam:25s}): ({x:.3f}, {y:.3f})')

dists = cdist(avg_coeffs, avg_coeffs, metric='cosine')
print('\nNearest task by coefficient cosine similarity:')
for i, tid in enumerate(task_ids):
    nearest_idx = np.argsort(dists[i])[1]
    print(f'  {tid:15s} -> {task_ids[nearest_idx]:15s} (dist={dists[i][nearest_idx]:.3f}, same_family={families[i]==families[nearest_idx]})')

results = {
    'pca_coords': {tid: coords[i].tolist() for i, tid in enumerate(task_ids)},
    'families': {tid: fam for tid, fam in zip(task_ids, families)},
    'nearest_task': {tid: task_ids[np.argsort(dists[i])[1]] for i, tid in enumerate(task_ids)},
    'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
}
print('\n=== RESULTS ===')
print(json.dumps(results, indent=2))
print('=== END ===')
