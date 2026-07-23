"""Evaluate frozen backbone on 20 held-out SNI tasks."""
import sys, json, torch, numpy as np
sys.path.insert(0, '.')
from models.backbone import load_backbone
from data.task_collection import HOLDOUT_TASKS_SNI, load_sni_dataset, filter_sni_tasks, format_sni_examples
from eval import evaluate_task
from train import set_seed

set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Loading backbone...')
model, tokenizer = load_backbone('Qwen/Qwen2.5-1.5B')
model = model.to(device)
model.eval()

print('Loading SNI dataset...')
full_ds = load_sni_dataset()
raw = filter_sni_tasks(full_ds, HOLDOUT_TASKS_SNI, 2000)

scores = {}
for tid in HOLDOUT_TASKS_SNI:
    if tid not in raw:
        print(f'  {tid}: NOT FOUND')
        continue
    examples = format_sni_examples(tid, raw[tid], 'validation')
    if not examples:
        continue
    if len(examples) > 200:
        examples = examples[:200]
    avg_len = np.mean([len(ex['response'].split()) for ex in examples[:20]])
    metric = 'accuracy' if avg_len < 5 else 'rouge_l'
    max_tok = 32 if avg_len < 5 else 128
    s = evaluate_task(model, tokenizer, tid, examples, metric, max_tok, device)
    scores[tid] = s
    print(f'  {tid}: {s}')

print('\n=== FROZEN BASELINE RESULTS ===')
print(json.dumps(scores, indent=2, default=str))
print('=== END ===')
