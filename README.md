# TaskMap

Code for the anonymous submission:

**TaskMap: Task-Conditioned LLM Adaptation in a Shared Orthogonal Basis**

TaskMap is a parameter-efficient adaptation method for heterogeneous language-model tasks. It represents task-specific feed-forward-network (FFN) updates using compact coefficient vectors in shared fixed orthogonal bases. A task-conditioned mapper generates block-specific coefficients, while the frozen backbone remains unchanged. An optional router can select a subset of blocks to receive task-specific residual updates.

Our ablations show that the main benefit comes from the task-conditioned coefficient values rather than the routing decisions themselves: destroying or sharing the coefficients substantially degrades performance, whereas applying residual updates to all blocks or randomizing the routing does not.

A key property of the representation is that, for a fixed layer, block, and projection,

$$
\Delta W(c) = A\,\mathrm{diag}(c)\,B
$$

with orthonormal columns of $A$ and orthonormal rows of $B$, gives

$$
\langle \Delta W(c), \Delta W(d) \rangle_F = c^\top d
$$

and

$$
\| \Delta W(c) - \Delta W(d) \|_F = \| c - d \|_2
$$

Thus the coefficient vector provides an identifiable, distance-preserving coordinate representation of the corresponding representable weight update. This contrasts with conventional LoRA factorizations, whose factors are generally non-unique.

Experiments use **Qwen2.5-1.5B** across classification, question answering, summarization, translation, mathematical reasoning, and cold-start task generalization.

## Main Findings

- **Representation:** shared fixed orthogonal bases provide a common low-dimensional coordinate system for task adaptations.
- **Mechanism:** task-conditioned, block-specific coefficient values are the main source of TaskMap's gains; learned routing and sparsity are not necessary for the observed accuracy improvements.
- **Generalization:** task descriptions support zero-shot coefficient generation for unseen tasks, but robustness to description wording remains a primary limitation.

## Repository Structure

```text
TaskMap/
├── models/
│   ├── taskmap_model.py        # Main TaskMap model
│   ├── task_code.py            # Description embeddings + learned residual task codes
│   ├── mapper.py               # Task-conditioned mapper
│   ├── router.py               # Optional top-k block routing
│   ├── block_residuals.py      # Fixed-basis low-rank residuals
│   ├── ffn_hooks.py            # Additive FFN residual injection
│   ├── baselines.py            # Additional baseline modules
│   └── backbone.py             # Frozen backbone / PEFT helpers
├── data/
│   ├── config.py               # Known-task definitions
│   ├── task_collection.py      # 41-train / 20-held-out SNI task collection
│   ├── download.py             # Dataset loading helpers
│   ├── format.py               # Prompt / response formatting
│   └── sampler.py              # Task-homogeneous microbatch sampling
├── configs/
│   ├── taskmap_no_topology.yaml
│   ├── taskmap_reference.yaml
│   └── baseline_lora.yaml
├── analysis/
│   ├── route_analysis.py
│   ├── route_overlap.py
│   └── neuron_clustering.py
├── train.py                    # Frozen / shared LoRA / VeRA baselines
├── train_taskmap.py            # TaskMap on known tasks
├── train_taskmap_scaled.py     # TaskMap on 41 SNI tasks + cold-start evaluation
├── train_per_task_lora.py      # Per-task LoRA baseline
├── train_direct_block_lora.py  # Direct coefficient optimization baseline
├── train_hyperlora.py          # Full-Factor Hypernetwork on known tasks
├── train_hyperlora_sni.py      # Full-Factor Hypernetwork on SNI cold-start
├── train_lora_sni41.py         # Shared LoRA on the 41 SNI training tasks
├── train_ablations.py          # Coefficient / mapper / layer ablations
├── eval.py                     # Log-likelihood classification + generation metrics
├── eval_coldstart_baselines.py
├── eval_frozen_sni_holdout.py
├── extract_coefficients.py     # PCA / nearest-neighbor coefficient analysis
├── losses.py
└── requirements.txt
```

## Setup

```bash
conda create -n taskmap python=3.10 -y
conda activate taskmap
pip install -r requirements.txt
```

The reported experiments use **Qwen2.5-1.5B** and one NVIDIA **H100 80GB** GPU. Smaller-memory setups may require a reduced microbatch size and increased gradient accumulation.

## Reproducing the Reported Experiments

### 1. Known-Task TaskMap — Paper Table 2

The main known-task experiments evaluate 9 tasks and use 6,000 training steps.

```bash
python train_taskmap.py \
  --config configs/taskmap_no_topology.yaml \
  --backbone Qwen/Qwen2.5-1.5B \
  --active_fraction 0.75 \
  --unfreeze_mapper \
  --max_steps 6000 \
  --seed 42 \
  --paper_tasks \
  --save_checkpoint
```

Repeat with seeds `137` and `2024` for the reported three-seed TaskMap result.

### 2. Known-Task Baselines — Paper Table 2

Shared LoRA, rank 8:

```bash
python train.py \
  --mode lora \
  --backbone Qwen/Qwen2.5-1.5B \
  --lora_rank 8 \
  --max_steps 6000 \
  --seed 42 \
  --paper_tasks \
  --output_dir outputs/lora_r8_seed42
```

Capacity-matched LoRA, rank 40:

```bash
python train.py \
  --mode lora \
  --backbone Qwen/Qwen2.5-1.5B \
  --lora_rank 40 \
  --max_steps 6000 \
  --seed 42 \
  --paper_tasks \
  --output_dir outputs/lora_r40_seed42
```

VeRA, rank 256:

```bash
python train.py \
  --mode vera \
  --backbone Qwen/Qwen2.5-1.5B \
  --lora_rank 256 \
  --max_steps 6000 \
  --seed 42 \
  --paper_tasks \
  --output_dir outputs/vera_r256_seed42
```

Direct optimization:

```bash
python train_direct_block_lora.py \
  --backbone Qwen/Qwen2.5-1.5B \
  --max_steps 6000 \
  --seed 42 \
  --paper_tasks
```

Full-Factor Hypernetwork:

```bash
python train_hyperlora.py \
  --backbone Qwen/Qwen2.5-1.5B \
  --max_steps 6000 \
  --seed 42 \
  --paper_tasks
```

Per-task LoRA is implemented in:

```text
train_per_task_lora.py
```

### 3. Cold-Start TaskMap — Paper Table 3

TaskMap is trained on 41 Super-NaturalInstructions tasks and evaluated without task-specific training on 20 held-out tasks. The reported TaskMap cold-start runs use 12,000 training steps.

```bash
python train_taskmap_scaled.py \
  --config configs/taskmap_no_topology.yaml \
  --backbone Qwen/Qwen2.5-1.5B \
  --max_steps 12000 \
  --active_fraction 0.75 \
  --unfreeze_mapper \
  --seed 42 \
  --max_per_task 2000 \
  --max_eval_examples 200 \
  --output_dir outputs/taskmap_coldstart_s42
```

Repeat with seeds `137` and `2024`.

At cold-start, the learned per-task residual code is unavailable, so TaskMap generates the adaptation from the task description alone.

### 4. Cold-Start Baselines — Paper Tables 3–4

Shared LoRA on the same 41 SNI training tasks:

```bash
python train_lora_sni41.py \
  --backbone Qwen/Qwen2.5-1.5B \
  --max_steps 6000 \
  --seed 42
```

Full-Factor Hypernetwork:

```bash
python train_hyperlora_sni.py \
  --backbone Qwen/Qwen2.5-1.5B \
  --max_steps 12000 \
  --seed 42
```

Frozen and nearest-reuse cold-start baselines are implemented in:

```text
eval_frozen_sni_holdout.py
eval_coldstart_baselines.py
```

### 5. Ablations — Paper Table 5

Examples:

```bash
# Shared coefficients across blocks
python train_ablations.py --ablation shared_coefficients

# Linear mapper
python train_ablations.py --ablation linear_mapper

# Layer-selection ablations
python train_ablations.py --ablation top_layers_only
python train_ablations.py --ablation middle_layers_only
python train_ablations.py --ablation bottom_layers_only
```

Semantic ablations are exposed directly by `train_taskmap.py`:

```bash
# Shuffle task descriptions
python train_taskmap.py \
  --config configs/taskmap_no_topology.yaml \
  --unfreeze_mapper \
  --max_steps 6000 \
  --shuffle_descriptions \
  --seed 42

# Replace descriptions with task-ID vectors
python train_taskmap.py \
  --config configs/taskmap_no_topology.yaml \
  --unfreeze_mapper \
  --max_steps 6000 \
  --task_id_conditioning \
  --seed 42

# Description-only conditioning: remove learned residual codes
python train_taskmap.py \
  --config configs/taskmap_no_topology.yaml \
  --unfreeze_mapper \
  --max_steps 6000 \
  --no_residual \
  --seed 42
```

### What Matters in TaskMap?

Ablations show that TaskMap's performance is primarily driven by
task-conditioned block-specific coefficient values rather than sparse routing.

| Variant | Change from TaskMap |
|---|---:|
| Random coefficients | -15.8 |
| Shared coefficient per layer | -4.8 |
| Without balance loss | +0.0 |
| All blocks (`rho = 1.0`) | +0.4 |
| Random routing | +2.2 |

These results indicate that coefficient specialization is the key mechanism:
destroying or sharing coefficients substantially hurts performance, whereas
removing sparsity or changing the routing policy does not.

### 6. Coefficient-Space Analysis — Paper Figure 4

Train TaskMap with checkpoint saving enabled, then run:

```bash
python extract_coefficients.py
```

This extracts learned task coefficients, computes the PCA visualization, and reports nearest tasks by coefficient cosine similarity.

The exact coefficient-space isometry applies within each fixed layer/block/projection basis. Figure 4 averages coefficients across layers before PCA and should therefore be interpreted as an empirical summary of the shared coordinate structure.

## Main Results

### Known Tasks — Paper Table 2

| Method | Trainable Parameters | Macro |
|---|---:|---:|
| Frozen base | 0 | 31.0 |
| LoRA r=40 (capacity matched) | 35.3M | 37.0 |
| Shared LoRA r=8 | 7.1M | 40.3 ± 1.0 |
| Per-task LoRA r=8 | 7.1M / task | 40.6 |
| VeRA r=256 | 0.6M | 42.5 |
| **TaskMap** | **35.7M** | **46.3 ± 2.2** |
| Full-Factor Hypernetwork | 2.9M | 47.7 |
| Direct Optimization | 490K total | 49.3 ± 0.9 |

The macro score averages heterogeneous task metrics on a common 0–100 scale and is used as a compact summary; task-level values should be interpreted within metric type.

### Cold-Start — Paper Tables 3–4

| Method | Macro | W / T / L vs. Frozen | Median Δ | Worst Δ |
|---|---:|---:|---:|---:|
| Frozen | 14.7 | — | — | — |
| Shared LoRA | **24.0 ± 0.1** | 14 / 1 / 5 | +3.8 | -26.7 |
| Full-Factor Hypernetwork | 21.5 ± 0.9 | 11 / 6 / 3 | +3.9 | **-4.2** |
| **TaskMap** | **20.3 ± 1.4** | **11 / 6 / 3** | **+4.8** | -7.9 |
| Nearest reuse | 18.6 ± 2.3 | 8 / 5 / 7 | +0.0 | -18.6 |

Shared LoRA attains the highest mean. Task-conditioned generation gives a more conservative transfer profile: TaskMap and the Full-Factor Hypernetwork each have only three negative-transfer tasks, versus five for Shared LoRA, and TaskMap has the largest median improvement.

## Evaluation Protocol

- **Classification:** first-token log-probability over the allowed labels.
- **Question answering:** F1.
- **Summarization:** ROUGE-L.
- **Translation:** sacreBLEU.
- **Mathematical reasoning:** exact answer.
- The nine-task macro averages heterogeneous metrics on a 0–100 scale; comparisons are most meaningful within metric type.

## Efficiency Notes

For Qwen2.5-1.5B, TaskMap uses approximately **35.7M shared trainable parameters** plus **896 learned residual parameters per observed task**.

The fixed orthogonal bases total approximately **313 MB**, but they are regenerable from seeds and therefore do not need to be persisted.

The current implementation injects block residuals through Python forward hooks and is intended for research evaluation rather than optimized inference throughput.

## Hardware

Reported experiments use one NVIDIA H100 80GB GPU.

Representative training times:

- Shared LoRA, 6K steps: approximately 1.2 hours.
- Full-Factor Hypernetwork, 6K steps: approximately 1.2 hours.
- TaskMap, 6K known-task run: approximately 8.7 hours.

Actual runtime depends on hardware, software versions, dataset caching, and evaluation frequency.

## Anonymity

This repository accompanies an anonymous conference submission. Author names, affiliations, acknowledgments, personal usernames, and identifying external links should not be added to the anonymized review mirror.

For double-blind review, link only to the anonymized mirror of this repository rather than to the original GitHub repository.
