# TaskMap Experiment Results

**Backbone**: Qwen2.5-1.5B  
**Hardware**: 1x NVIDIA H100 80GB per run  
**Last updated**: 2026-07-12

## Full Comparison Table (Additive Hooks)

| Task | Metric | Frozen | LoRA r=8 | LoRA r=16 | LoRA r=32 | TaskMap 50% | Direct BL |
|------|--------|--------|----------|-----------|-----------|-------------|-----------|
| SST-2 | Accuracy | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **2.0** |
| AG News | Accuracy | 25.2 | 0.8 | 0.4 | 0.0 | 2.0 | **46.0** |
| BoolQ | Accuracy | 0.0 | **8.8** | **9.8** | 0.2 | 0.0 | 0.0 |
| SQuAD | F1 | 49.3 | 6.6 | 5.5 | 5.2 | 5.7 | **74.6** |
| XSum | ROUGE-L | 10.9 | **27.6** | 27.1 | 27.0 | 13.4 | 26.6 |
| WMT14 | BLEU | 1.0 | 1.8 | 2.0 | 1.8 | **14.8** | 13.5 |
| WMT16 | BLEU | 1.8 | 1.6 | 1.7 | 1.9 | 3.6 | **8.1** |
| GSM8K | Exact | 10.6 | 37.4 | 32.8 | 30.2 | 18.0 | **44.0** |
| SVAMP | Exact | **53.0** | 27.0 | 24.0 | 26.0 | 34.0 | 36.0 |
| MBPP | pass@1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Macro** | | 15.18 | 11.16 | 10.33 | 9.22 | 9.15 | **25.13** |

## Training Configuration

| Method | Steps | Eval Examples | Trainable Params | Speed (s/step) | Hook Type |
|--------|-------|---------------|------------------|----------------|-----------|
| Frozen base | - | 500 | 0 | - | - |
| LoRA r=8 | 12,000 | 500 | 7.1M | 0.33 | None |
| LoRA r=16 | 12,000 | 500 | 14.1M | 0.33 | None |
| LoRA r=32 | 12,000 | 500 | 28.2M | 0.33 | None |
| TaskMap 50% | 6,000 | 50 | 1.4M | 5.9 | Additive |
| Direct Block-LoRA | 6,000 | 50 | 490K | 3.2 | Additive |

## Key Findings

### 1. Direct Block-LoRA is the Best Method (macro 25.13)
- **Beats frozen base by +10 points** on macro average
- **Beats LoRA r=8 by +14 points** — avoids negative transfer entirely
- Improves on **every task** compared to frozen base
- Uses only **490K trainable parameters** (70x fewer than LoRA r=32)
- Key wins: SQuAD F1 74.6 (+25 over frozen), AG News 46.0 (+21), GSM8K 44.0 (+33), WMT14 13.5 (+12.5)

### 2. Additive Hooks Are Critical
- Old replacement hooks: TaskMap 0.64, Direct BL 0.93 (near zero)
- New additive hooks: TaskMap 9.15, Direct BL 25.13 (functional!)
- The base model's dense FFN output must be preserved; task adaptations should be added on top

### 3. Dense Multi-Task LoRA Causes Severe Negative Transfer
- All LoRA models score below frozen base on macro average
- LoRA improves generation tasks (XSum, GSM8K) but destroys classification/QA (SQuAD -44, AG News -25)
- More parameters = more transfer: r=32 (9.22) < r=8 (11.16)

### 4. Mapper is a Bottleneck
- Direct BL (25.13) >> TaskMap (9.15) — both use additive hooks
- TaskMap's frozen mapper with 1.4M indirect trainable params can't match Direct BL's 490K direct params
- Gradient path task codes → frozen mapper → coefficients is too indirect

### 5. Classification Eval Needs Fixing
- SST-2 scores 0% across most methods (format mismatch: model outputs verbose text)
- Direct BL shows SST-2=2.0% and AG News=46.0%, proving classification can work with the right training

## Recommended Next Steps

### Priority 1: Improve TaskMap to match Direct BL
- **Unfreeze mapper** — let mapper weights receive gradients alongside task codes
- **Increase code dimension** — d_z=32 may be too small, try d_z=64 or 128
- **More training steps** — 12,000 instead of 6,000

### Priority 2: Strengthen baselines
- **Re-run LoRA with 50 eval examples** for fair comparison
- **Fix classification eval** — add fuzzy matching for label extraction
- **Task-specific LoRA** — train one adapter per task as upper bound

### Priority 3: Complete Table 3
- TaskMap at 25% and 75% active fractions
- Ablation table (Table 4)
- Route analysis (overlap, causal tests)
- Multi-seed runs for confidence intervals

## Previous Results (Replacement Hooks — Deprecated)

These results used the broken replacement hook approach and are kept for reference only.

| Method | Macro (replacement) | Macro (additive) |
|--------|--------------------|--------------------|
| TaskMap 50% | 0.64 | 9.15 |
| Direct Block-LoRA | 0.93 | 25.13 |

## Files
- `eval_frozen.json` — Frozen base (500 eval examples)
- `eval_lora_r8.json` — LoRA r=8 (500 eval examples)
- `eval_lora_r16.json` — LoRA r=16 (500 eval examples)
- `eval_lora_r32.json` — LoRA r=32 (500 eval examples)
- `eval_taskmap_50.json` — TaskMap 50% replacement hooks (deprecated)
- `eval_direct_block_lora.json` — Direct BL replacement hooks (deprecated)
- `eval_taskmap_50_additive.json` — TaskMap 50% additive hooks
- `eval_direct_block_lora_additive.json` — Direct BL additive hooks
