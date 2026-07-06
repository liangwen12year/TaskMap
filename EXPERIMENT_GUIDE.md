# TaskMap Experiment Guide

## Prerequisites

- GPU machine with at least 1x A100 40GB (or 2x consumer GPUs with 24GB each)
- Python 3.10+, CUDA 12.x, PyTorch 2.2+
- Clone/copy the TaskMap directory to the GPU machine

## Setup

```bash
cd TaskMap
pip install -r requirements.txt
```

Verify installation:
```bash
python test_data_pipeline.py
python test_taskmap_arch.py
python test_taskmap_train.py
python test_step5.py
```

All 4 test suites should pass.

---

## Phase 1: Baselines (Table 3, rows 1-3)

These establish the scores TaskMap must match or beat.

### 1.1 Frozen base (no adaptation)

```bash
python run_eval.py --mode frozen \
    --backbone Qwen/Qwen2.5-1.5B \
    --output_file outputs/eval_frozen.json \
    --max_examples 500
```

Expected: low scores across all tasks (this is the floor).
Time: ~30 min on 1x A100.

### 1.2 Dense multi-task LoRA (primary baseline)

Train at three ranks to find the best:

```bash
# Rank 8
python train.py --mode lora --backbone Qwen/Qwen2.5-1.5B \
    --lora_rank 8 --max_steps 12000 \
    --output_dir outputs/lora_r8 --seed 42

# Rank 16 (reference)
python train.py --mode lora --backbone Qwen/Qwen2.5-1.5B \
    --lora_rank 16 --max_steps 12000 \
    --output_dir outputs/lora_r16 --seed 42

# Rank 32
python train.py --mode lora --backbone Qwen/Qwen2.5-1.5B \
    --lora_rank 32 --max_steps 12000 \
    --output_dir outputs/lora_r32 --seed 42
```

Time: ~4-6 hours each on 1x A100.

Evaluate each:

```bash
python run_eval.py --mode lora --backbone Qwen/Qwen2.5-1.5B \
    --checkpoint outputs/lora_r16/final \
    --lora_rank 16 \
    --output_file outputs/eval_lora_r16.json \
    --max_examples 500
```

Pick the rank with the best macro score. This is the target TaskMap must match.

### 1.3 Verify results before proceeding

Check outputs/eval_*.json files. The frozen base should score much lower than LoRA. If LoRA scores are unreasonably low, debug before continuing (likely a data or formatting issue).

---

## Phase 2: TaskMap Training (Table 3, rows 8-10)

### 2.1 TaskMap at 50% (reference configuration)

```bash
python train_taskmap.py \
    --config configs/taskmap_reference.yaml \
    --backbone Qwen/Qwen2.5-1.5B \
    --seed 42
```

Time: ~6-8 hours on 1x A100.

Monitor during training:
- Loss should decrease steadily
- After warmup (step ~360), routes should stabilize
- Budget loss should approach 0 (gates matching target fraction)

### 2.2 TaskMap at 25% and 75%

Edit active_fraction in the config or pass overrides:

```bash
# 25% active
python train_taskmap.py \
    --config configs/taskmap_reference.yaml \
    --backbone Qwen/Qwen2.5-1.5B \
    --output_dir outputs/taskmap_25 \
    --seed 42
# (manually edit configs/taskmap_reference.yaml to set active_fraction: 0.25
#  or create configs/taskmap_25.yaml with that change)

# 75% active
python train_taskmap.py \
    --config configs/taskmap_reference.yaml \
    --backbone Qwen/Qwen2.5-1.5B \
    --output_dir outputs/taskmap_75 \
    --seed 42
# (set active_fraction: 0.75)
```

### 2.3 Evaluate TaskMap

```bash
python run_eval.py --mode taskmap \
    --backbone Qwen/Qwen2.5-1.5B \
    --checkpoint outputs/taskmap_reference/final \
    --output_file outputs/eval_taskmap_50.json \
    --max_examples 500
```

---

## Phase 3: Additional Baselines (Table 3, rows 4-7)

Run these after Phase 2 to complete the comparison.

### 3.1 Task-specific LoRA (row 4)

Train one LoRA adapter per task (12 separate training runs):

```bash
for TASK in sst2 agnews boolq squad xsum samsum wmt14_ende wmt16_enro gsm8k svamp apps_intro mbpp; do
    python train.py --mode lora --backbone Qwen/Qwen2.5-1.5B \
        --lora_rank 16 --max_steps 12000 \
        --output_dir outputs/lora_specific_${TASK} \
        --seed 42
    # Note: modify train.py or create a wrapper to filter to single task
done
```

This requires a small modification to train.py to accept a --task_filter argument. The task-specific scores serve as the upper bound for computing negative transfer I_t.

### 3.2 Task-family LoRA (row 5)

Train one adapter per family (6 runs):

```bash
for FAMILY in classification question_answering summarization translation mathematical_reasoning code_generation; do
    python train.py --mode lora --backbone Qwen/Qwen2.5-1.5B \
        --lora_rank 16 --max_steps 12000 \
        --output_dir outputs/lora_family_${FAMILY} \
        --seed 42
    # Note: filter training data to tasks in this family only
done
```

### 3.3 Direct Block-LoRA (row 6)

Requires a training script variant that uses DirectBlockLoRA instead of the mapper. Reuse train_taskmap.py structure but replace the TaskMap model with DirectBlockLoRA from models/baselines.py.

### 3.4 Random-route mapped (row 7)

Same as TaskMap training but routes are fixed random (not learned). Use RandomRouteTaskMap from models/baselines.py.

---

## Phase 4: Generate Table 3

After all experiments complete:

```bash
python generate_table3.py --results_dir outputs/
```

This reads all eval_*.json files and prints the formatted table.

### Success criteria (from paper Section 4.1):

1. TaskMap macro score within 1 absolute point of the best dense LoRA
2. Mean negative transfer I_t is no worse than dense LoRA
3. Structured kernel improves tokens/s or peak memory by at least 10%

---

## Phase 5: Multiple Seeds (Tier A requires 3 seeds)

After finding the best configuration, rerun with seeds 42, 137, 2024:

```bash
for SEED in 42 137 2024; do
    # Best LoRA baseline
    python train.py --mode lora --backbone Qwen/Qwen2.5-1.5B \
        --lora_rank 16 --max_steps 12000 \
        --output_dir outputs/lora_r16_seed${SEED} --seed ${SEED}

    # TaskMap 50%
    python train_taskmap.py \
        --config configs/taskmap_reference.yaml \
        --output_dir outputs/taskmap_50_seed${SEED} --seed ${SEED}
done
```

Report mean, std, and 95% confidence intervals across seeds.

---

## Quick Reference: Run Order

| Priority | What | Command | Time (1x A100) |
|----------|------|---------|-----------------|
| 1 | Frozen base eval | `python run_eval.py --mode frozen ...` | 30 min |
| 2 | LoRA r=16 train | `python train.py --mode lora --lora_rank 16 ...` | 4-6 hr |
| 3 | LoRA r=16 eval | `python run_eval.py --mode lora ...` | 30 min |
| 4 | TaskMap 50% train | `python train_taskmap.py --config ...` | 6-8 hr |
| 5 | TaskMap 50% eval | `python run_eval.py --mode taskmap ...` | 30 min |
| 6 | Compare scores | Check eval JSONs, decide if TaskMap is competitive | - |
| 7 | TaskMap 25%, 75% | Same as step 4 with different active_fraction | 6-8 hr each |
| 8 | Additional baselines | Task-specific, family, Direct Block-LoRA, random | 4-6 hr each |
| 9 | Multi-seed runs | Seeds 42, 137, 2024 for best config | 3x time |
| 10 | Generate Table 3 | `python generate_table3.py` | instant |

Total estimated time for full Table 3: ~80-100 GPU hours on 1x A100.

---

## Troubleshooting

### Out of memory
- Reduce microbatch_size to 2 or 1
- Increase gradient_accumulation_steps proportionally
- Enable gradient checkpointing: add `model.gradient_checkpointing_enable()` after loading

### Dataset download failures
4 datasets may fail (samsum, wmt14, wmt16, apps). Fix dataset paths in data/config.py:
- samsum: try `"samsum"` instead of `"Samsung/samsum"`
- wmt14: try `"wmt/wmt14"` or download manually
- wmt16: try `"wmt/wmt16"` or download manually
- apps: try `"BAAI/TACO"` or download APPS manually from GitHub

### Training loss not decreasing
- Check that backbone is frozen (no gradients on backbone params)
- Check learning rates (codes: 2e-3, projectors: 2e-4)
- Check that tokenization correctly masks prompt tokens with -100

### Routes all identical (route collapse)
- Check balance loss is active (lambda_bal > 0)
- Check topology loss target values (pi_near vs pi_far)
- Inspect route overlap with: `python -c "from analysis.route_overlap import *; ..."`
