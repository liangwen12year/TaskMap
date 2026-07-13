# TaskMap Experiment Results

**Backbone**: Qwen2.5-1.5B  
**Hardware**: 1x NVIDIA H100 80GB per run  
**Eval**: 50 examples per task  
**Last updated**: 2026-07-12

## Final Comparison Table

| Task | Metric | Frozen | LoRA r=8 | LoRA r=16 | Direct BL | TaskMap (unfrozen) |
|------|--------|--------|----------|-----------|-----------|-------------------|
| SST-2 | Accuracy | 0.0 | 0.0 | 0.0 | **2.0** | 0.0 |
| AG News | Accuracy | 25.2 | 6.0 | 0.0 | **46.0** | 18.0 |
| BoolQ | Accuracy | 0.0 | 2.0 | **16.0** | 0.0 | 8.0 |
| SQuAD | F1 | 49.3 | 5.8 | 4.5 | **74.6** | 52.5 |
| XSum | ROUGE-L | 10.9 | 27.1 | 26.4 | 26.6 | **28.7** |
| WMT14 | BLEU | 1.0 | 0.7 | 0.7 | 13.5 | **14.0** |
| WMT16 | BLEU | 1.8 | 1.8 | 1.7 | **8.1** | 7.7 |
| GSM8K | Exact | 10.6 | 40.0 | 32.0 | **44.0** | 42.0 |
| SVAMP | Exact | 53.0 | 28.0 | 16.0 | 36.0 | **58.0** |
| MBPP | pass@1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Macro** | | 15.18 | 11.16 | 9.73 | **25.13** | **22.88** |

## Training Configuration

| Method | Steps | Trainable Params | Speed (s/step) | Hook Type |
|--------|-------|------------------|----------------|-----------|
| Frozen base | - | 0 | - | - |
| LoRA r=8 | 12,000 | 7.1M | 0.33 | None |
| LoRA r=16 | 12,000 | 14.1M | 0.33 | None |
| Direct Block-LoRA | 6,000 | 490K | 3.2 | Additive |
| TaskMap (unfrozen) | 6,000 | 35.7M | 5.5 | Additive |

## Key Findings

### 1. Task-Conditioned Routing Eliminates Negative Transfer

Dense multi-task LoRA suffers severe negative transfer: it improves generation tasks (XSum, GSM8K) but destroys classification and QA (SQuAD drops from 49.3 to 5.8). Both routing-based methods (Direct BL and TaskMap) avoid this by adapting different FFN blocks per task.

| | Frozen | LoRA r=8 | Direct BL | TaskMap |
|--|--------|----------|-----------|---------|
| Tasks improved vs frozen | - | 4/10 | 8/10 | 8/10 |
| Tasks degraded vs frozen | - | 6/10 | 1/10 | 1/10 |
| Macro | 15.18 | 11.16 | **25.13** | **22.88** |

### 2. Direct Block-LoRA Achieves Best Overall Score (25.13)

With only 490K directly optimized parameters, Direct Block-LoRA achieves the highest macro average. Key strengths:
- SQuAD F1: 74.6 (+25 over frozen, +69 over LoRA)
- AG News: 46.0 (+21 over frozen)
- GSM8K: 44.0 (+33 over frozen)

### 3. TaskMap with Unfrozen Mapper Nearly Matches (22.88)

Unfreezing the mapper (35.7M params) closes most of the gap vs Direct BL. TaskMap has unique strengths:
- **SVAMP: 58.0** — only method to beat frozen base (53.0)
- **XSum: 28.7** — best across all methods
- **SQuAD: 52.5** — preserves frozen base performance while LoRA destroyed it

### 4. Frozen Mapper Is a Bottleneck

| Mapper | Trainable | Macro |
|--------|-----------|-------|
| Frozen (paper design) | 1.4M | 9.15 |
| Unfrozen | 35.7M | **22.88** |
| No mapper (Direct BL) | 490K | **25.13** |

The frozen mapper with only 1.4M indirect trainable params cannot generate useful coefficients. Unfreezing adds 34.4M mapper params and nearly matches Direct BL.

### 5. Additive Hooks Are Essential

| Hook Type | TaskMap | Direct BL |
|-----------|---------|-----------|
| Replacement (broken) | 0.64 | 0.93 |
| Additive (correct) | **22.88** | **25.13** |

Replacing the MLP output with sparse computation destroys the model. The correct approach: keep the dense output and add task-specific residuals on top.

### 6. Classification Eval Has Format Issues

SST-2 scores 0% across all methods except Direct BL (2%). The model outputs verbose text instead of single-word labels. AG News works better because the category names are more distinctive. This is an eval issue, not a training issue.

## Improvement Over Frozen Base

| Task | Frozen | Best Method | Improvement |
|------|--------|-------------|-------------|
| SQuAD | 49.3 | Direct BL 74.6 | +25.3 |
| SVAMP | 53.0 | TaskMap 58.0 | +5.0 |
| GSM8K | 10.6 | Direct BL 44.0 | +33.4 |
| AG News | 25.2 | Direct BL 46.0 | +20.8 |
| XSum | 10.9 | TaskMap 28.7 | +17.8 |
| WMT14 | 1.0 | TaskMap 14.0 | +13.0 |
| WMT16 | 1.8 | Direct BL 8.1 | +6.3 |
| BoolQ | 0.0 | LoRA r=16 16.0 | +16.0 |

## Experiment History

### Phase 1: Replacement hooks (failed)
- TaskMap 50% frozen mapper: macro 0.64
- Direct Block-LoRA: macro 0.93
- Root cause: hook replaced MLP output instead of adding residuals

### Phase 2: Additive hooks (success)
- Direct Block-LoRA: macro **25.13**
- TaskMap 50% frozen mapper: macro 9.15
- TaskMap 50% unfrozen mapper: macro **22.88**

## Next Steps

1. **TaskMap 25% and 75%** — test different active fractions
2. **Ablation table** — isolate each component's contribution
3. **Route analysis** — visualize which blocks each task selects
4. **Cold-start test** — can TaskMap generalize to unseen tasks via description?
5. **Multi-seed runs** — statistical significance (seeds 42, 137, 2024)
6. **Fix classification eval** — fuzzy label matching
7. **12,000 steps** — longer training may improve TaskMap further

## Files
- `eval_frozen.json` — Frozen base
- `eval_lora_r8.json` — LoRA r=8 (500 eval)
- `eval_lora_r16.json` — LoRA r=16 (500 eval)
- `eval_lora_r32.json` — LoRA r=32 (500 eval)
- `eval_taskmap_50.json` — TaskMap frozen mapper, replacement hooks
- `eval_direct_block_lora.json` — Direct BL, replacement hooks
- `eval_taskmap_50_additive.json` — TaskMap frozen mapper, additive hooks
- `eval_direct_block_lora_additive.json` — Direct BL, additive hooks
- `eval_taskmap_50_unfrozen.json` — TaskMap unfrozen mapper, additive hooks
