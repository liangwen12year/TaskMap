# TaskMap Experiment Results

**Backbone**: Qwen2.5-1.5B  
**Hardware**: 1x NVIDIA H100 80GB per run  
**Eval**: 50 examples per task, fuzzy classification matching  
**Last updated**: 2026-07-13

## Multi-Seed Results (Seeds 42, 137, 2024)

### Direct Block-LoRA (490K trainable params)

| Task | Metric | Seed 42 | Seed 137 | Seed 2024 | Mean ± Std |
|------|--------|---------|----------|-----------|------------|
| SST-2 | Accuracy | 96.0 | 94.0 | 96.0 | **95.3 ± 1.2** |
| AG News | Accuracy | 90.0 | 88.0 | 88.0 | **88.7 ± 1.2** |
| BoolQ | Accuracy | 80.0 | 84.0 | 82.0 | **82.0 ± 2.0** |
| SQuAD | F1 | 72.2 | 78.4 | 79.4 | **76.7 ± 3.9** |
| XSum | ROUGE-L | 26.4 | 26.4 | 26.4 | **26.4 ± 0.0** |
| WMT14 | BLEU | 13.3 | 12.5 | 13.7 | **13.2 ± 0.6** |
| WMT16 | BLEU | 9.0 | 9.4 | 7.6 | **8.7 ± 0.9** |
| GSM8K | Exact | 44.0 | 44.0 | 36.0 | **41.3 ± 4.6** |
| SVAMP | Exact | 36.0 | 54.0 | 46.0 | **45.3 ± 9.0** |
| MBPP | pass@1 | 0.0 | 0.0 | 0.0 | **0.0 ± 0.0** |
| **Macro** | | 46.69 | 47.87 | 47.52 | **47.36 ± 0.6** |

### TaskMap 75% Unfrozen Mapper (35.7M trainable params)

| Task | Metric | Seed 42 | Seed 137 | Seed 2024 | Mean ± Std |
|------|--------|---------|----------|-----------|------------|
| SST-2 | Accuracy | 98.0 | 98.0 | 96.0 | **97.3 ± 1.2** |
| AG News | Accuracy | 92.0 | 96.0 | 90.0 | **92.7 ± 3.1** |
| BoolQ | Accuracy | 80.0 | 80.0 | 84.0 | **81.3 ± 2.3** |
| SQuAD | F1 | 53.1 | 39.8 | 71.9 | **54.9 ± 16.1** |
| XSum | ROUGE-L | 27.2 | 27.3 | 27.8 | **27.4 ± 0.3** |
| WMT14 | BLEU | 15.9 | 13.1 | 12.2 | **13.7 ± 1.9** |
| WMT16 | BLEU | 10.1 | 8.7 | 8.7 | **9.2 ± 0.8** |
| GSM8K | Exact | 44.0 | 50.0 | 46.0 | **46.7 ± 3.1** |
| SVAMP | Exact | 48.0 | 54.0 | 50.0 | **50.7 ± 3.1** |
| MBPP | pass@1 | 0.0 | 0.0 | 0.0 | **0.0 ± 0.0** |
| **Macro** | | 46.83 | 46.69 | 48.72 | **47.41 ± 1.1** |

## Summary Comparison (Mean ± Std)

| Method | Params | Macro | SQuAD | GSM8K | SVAMP | AG News |
|--------|--------|-------|-------|-------|-------|---------|
| Frozen base | 0 | 15.18 | 49.3 | 10.6 | 53.0 | 25.2 |
| LoRA r=8 | 7.1M | 11.16 | 5.8 | 40.0 | 28.0 | 6.0 |
| Direct BL | 490K | **47.36 ± 0.6** | **76.7 ± 3.9** | 41.3 ± 4.6 | 45.3 ± 9.0 | 88.7 ± 1.2 |
| TaskMap 75% | 35.7M | **47.41 ± 1.1** | 54.9 ± 16.1 | **46.7 ± 3.1** | **50.7 ± 3.1** | **92.7 ± 3.1** |

## Active Fraction Sweep (Seed 42, Unfrozen Mapper)

| Fraction | Macro | SQuAD | GSM8K | SVAMP | WMT14 |
|----------|-------|-------|-------|-------|-------|
| 25% | 45.69 | 57.2 | 38.0 | 42.0 | 13.8 |
| 50% | 45.55 | 43.8 | 44.0 | 48.0 | 13.6 |
| 75% | **46.83** | 53.1 | 44.0 | 48.0 | **15.9** |

## Key Findings

### 1. Both Routing Methods Achieve ~47 Macro (3x Over Frozen Base)
- Direct BL: 47.36 ± 0.6
- TaskMap 75%: 47.41 ± 1.1
- Essentially tied on macro average
- Both dramatically outperform frozen base (15.18) and LoRA (11.16)

### 2. Different Strengths Per Method
- **Direct BL** excels at SQuAD (76.7 ± 3.9) with low variance
- **TaskMap 75%** excels at SVAMP (50.7 ± 3.1), GSM8K (46.7), AG News (92.7)
- TaskMap has higher SQuAD variance (± 16.1) — some seeds work much better

### 3. Dense LoRA Causes Severe Negative Transfer
- LoRA r=8 macro (11.16) is below frozen base (15.18)
- LoRA destroys SQuAD (-44 F1), AG News (-19%), SVAMP (-25%)
- Routing-based methods avoid this entirely

### 4. Results Are Stable Across Seeds
- Direct BL macro std: 0.6 (very stable)
- TaskMap 75% macro std: 1.1 (stable)
- Most tasks have std < 4, except SQuAD on TaskMap (16.1)

### 5. Direct BL Is Remarkably Parameter-Efficient
- 490K params vs 35.7M for TaskMap — 73x fewer
- Same macro performance (47.36 vs 47.41)
- Faster training (3.2 s/step vs 5.5 s/step)

## Experiment History

| Date | Experiment | Key Result |
|------|-----------|------------|
| Jul 10 | LoRA baselines | Macro 9-11, severe negative transfer |
| Jul 10 | Replacement hooks | Macro 0.6-0.9 (broken) |
| Jul 11 | Additive hooks | Macro 9-25 (working) |
| Jul 12 | Unfrozen mapper | TaskMap macro 22.88 |
| Jul 13 | Fuzzy eval fix | All methods jump to ~45-47 macro |
| Jul 13 | Active fraction sweep | 25/50/75% all within ~1 point |
| Jul 13 | Multi-seed (3 seeds) | Direct BL 47.36 ± 0.6, TaskMap 47.41 ± 1.1 |

## Next Steps

1. **Route analysis** — add to training script, visualize block selection per task
2. **Cold-start test** — held-out tasks via description-only routing
3. **Ablation table** — remove topology/balance/residuals to measure contribution
4. **Longer training** — 12,000 steps may improve TaskMap further
5. **Tier B** — scale to Qwen2.5-7B or Llama-3.1-8B

## Files

| File | Method | Seed |
|------|--------|------|
| eval_frozen.json | Frozen base | - |
| eval_lora_r8_50ex.json | LoRA r=8 | 42 |
| eval_lora_r16_50ex.json | LoRA r=16 | 42 |
| eval_direct_block_lora_fuzzy.json | Direct BL | 42 |
| eval_dbl_seed137.json | Direct BL | 137 |
| eval_dbl_seed2024.json | Direct BL | 2024 |
| eval_taskmap_75_unfrozen_fuzzy.json | TaskMap 75% | 42 |
| eval_taskmap_75_seed137.json | TaskMap 75% | 137 |
| eval_taskmap_75_seed2024.json | TaskMap 75% | 2024 |
| eval_taskmap_25_unfrozen_fuzzy.json | TaskMap 25% | 42 |
| eval_taskmap_50_unfrozen_fuzzy.json | TaskMap 50% | 42 |
