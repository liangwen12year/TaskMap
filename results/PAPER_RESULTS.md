# Paper-result provenance

This directory contains outputs from both reported and exploratory experiments.
For the anonymous WSDM submission, use the paper and the top-level `README.md`
as the reviewer-facing numerical summary.

## Source-of-truth hierarchy

1. Executable code defines implementation semantics.
2. Completed seed-specific logs/results define reported experimental numbers.
3. Aggregate JSON summaries are convenience artifacts only and should not
   override completed seed-specific runs when an older aggregate is stale.
4. `data/config.py` pins the paper's 9-task evaluation set and the separate
   12-task Figure 4 analysis set.

## Verified canonical artifacts

- `eval_phase2_lora_coldstart_3seed.json`
  - matched Shared LoRA cold-start macros: 23.85, 24.03, 24.22
  - aggregate: 24.03 with std 0.19, reported as 24.0 ± 0.2
- `eval_exp2_random_coeff.json`
  - evaluation-only random-coefficient control used in the ablation discussion

The old `eval_phase1_multiseed.json` was removed from the reviewer-facing
branch because it was an incomplete historical aggregate: it retained a stale
TaskMap mean and marked the Direct Optimization seed-2024 run as unfinished
after that run had subsequently been completed.

Dated scratch-output folders `0712/`, `0713/`, and `0714/` were also removed
from the reviewer-facing branch. They remain recoverable from the
`wsdm-pre-cleanup-20260825` backup branch.

Historical or exploratory result files not listed above are retained when they
document useful negative results or auxiliary experiments. They should not be
treated as overriding the final paper/README values.
