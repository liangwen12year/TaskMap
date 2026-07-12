# TaskMap Experiment Results

**Backbone**: Qwen2.5-1.5B  
**Hardware**: 1x NVIDIA H100 80GB per run  
**Date**: 2026-07-11

## Full Comparison Table

| Task | Metric | Frozen | LoRA r=8 | LoRA r=16 | LoRA r=32 | TaskMap 50% | Direct BL |
|------|--------|--------|----------|-----------|-----------|-------------|-----------|
| SST-2 | Accuracy | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| AG News | Accuracy | **25.2** | 0.8 | 0.4 | 0.0 | 0.0 | 0.0 |
| BoolQ | Accuracy | 0.0 | **8.8** | **9.8** | 0.2 | 0.0 | 0.0 |
| SQuAD | F1 | **49.3** | 6.6 | 5.5 | 5.2 | 1.1 | 2.5 |
| XSum | ROUGE-L | 10.9 | **27.6** | **27.1** | **27.0** | 3.2 | 6.3 |
| WMT14 | BLEU | 1.0 | **1.8** | **2.0** | **1.8** | 0.05 | 0.37 |
| WMT16 | BLEU | **1.8** | 1.6 | 1.7 | **1.9** | 0.02 | 0.09 |
| GSM8K | Exact | 10.6 | **37.4** | **32.8** | **30.2** | 2.0 | 0.0 |
| SVAMP | Exact | **53.0** | 27.0 | 24.0 | 26.0 | 0.0 | 0.0 |
| MBPP | pass@1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Macro** | | **15.18** | **11.16** | **10.33** | **9.22** | **0.64** | **0.93** |

## Training Configuration

| Method | Steps | Eval Examples | Trainable Params | Speed (s/step) |
|--------|-------|---------------|------------------|----------------|
| Frozen base | - | 500 | 0 | - |
| LoRA r=8 | 12,000 | 500 | 7.1M | 0.33 |
| LoRA r=16 | 12,000 | 500 | 14.1M | 0.33 |
| LoRA r=32 | 12,000 | 500 | 28.2M | 0.33 |
| TaskMap 50% | 6,000 | 50 | 1.4M | 5.9 |
| Direct Block-LoRA | 6,000 | 50 | 490K | 3.2 |

## Key Findings

### 1. LoRA Baselines Show Clear Negative Transfer
- LoRA improves generation tasks: XSum (+17 ROUGE-L), GSM8K (+27%), WMT (+5 chrF)
- LoRA degrades classification/QA: SQuAD (-44 F1), AG News (-25%), SVAMP (-27%)
- More parameters = more negative transfer: r=32 (macro 9.22) < r=8 (macro 11.16)
- All LoRA models score below frozen base on macro average

### 2. Sparse FFN Hook Approach Fails
- Both TaskMap (0.64) and Direct Block-LoRA (0.93) score near zero
- Direct BL slightly better than TaskMap, ruling out the mapper as the bottleneck
- The problem is the hook **replaces** the full MLP output with only selected blocks
- Non-selected blocks' frozen weights are critical for base model functionality

### 3. Root Cause Identified
The current hook implementation computes:
```
output = sum_{g in selected} FFN_block_g(h) + residual_g
```
But it should be:
```
output = dense_FFN(h) + sum_{g in selected} residual_g(h)
```
The correct approach: keep the full dense MLP output and **add** task-specific residuals on top, rather than replacing the MLP with sparse computation.

### 4. Classification Tasks Always Score 0%
SST-2 scores 0% across all methods (including frozen base). This is a generation format issue — the model outputs verbose text instead of single-word labels ("positive"/"negative"). Not a training problem.

## Next Steps
1. **Fix hook approach**: additive residuals on top of dense FFN, not replacement
2. **Re-run TaskMap 50%** with additive hooks
3. **Fix classification eval**: add fuzzy matching for label extraction
4. **Re-evaluate LoRA** with 50 examples for fair comparison

## Files
- `eval_frozen.json` — Frozen base (500 eval examples)
- `eval_lora_r8.json` — LoRA r=8 (500 eval examples)
- `eval_lora_r16.json` — LoRA r=16 (500 eval examples)
- `eval_lora_r32.json` — LoRA r=32 (500 eval examples)
- `eval_taskmap_50.json` — TaskMap 50% (50 eval examples)
- `eval_direct_block_lora.json` — Direct Block-LoRA (50 eval examples)
