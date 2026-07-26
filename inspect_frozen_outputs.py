"""
Inspect frozen baseline outputs for SST-2 and BoolQ to diagnose
whether 0.0 accuracy is format failure or genuine inability.
Prints the first 20 predictions alongside references.
"""
import sys, torch
sys.path.insert(0, '.')
from models.backbone import load_backbone
from data.config import KNOWN_TASKS
from data.download import download_task
from data.format import format_all_tasks
from eval import generate_predictions
from train import set_seed

set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Loading Qwen2.5-1.5B...')
model, tokenizer = load_backbone('Qwen/Qwen2.5-1.5B')
model = model.to(device)
model.eval()

datasets = {}
for tid in ['sst2', 'boolq']:
    ds = download_task(tid, KNOWN_TASKS[tid])
    if ds:
        datasets[tid] = ds

eval_data = format_all_tasks(datasets, split='validation')

for tid in ['sst2', 'boolq']:
    examples = eval_data[tid][:20]
    preds = generate_predictions(model, tokenizer, examples,
                                 max_new_tokens=KNOWN_TASKS[tid]['max_response_tokens'],
                                 device=device)
    refs = [ex['response'] for ex in examples]

    print(f'\n=== {tid} (first 20) ===')
    match = 0
    for i, (p, r) in enumerate(zip(preds, refs)):
        fuzzy = r.lower() in p.lower()
        if fuzzy: match += 1
        print(f'  [{i:2d}] ref={r:10s} pred="{p[:80]}" {"✓" if fuzzy else "✗"}')
    print(f'  Fuzzy accuracy: {match}/{len(preds)} = {match/len(preds)*100:.0f}%')

print('\n=== Log-likelihood scoring ===')
from eval import classify_by_likelihood, CLASSIFICATION_LABELS
for tid in ['sst2', 'boolq']:
    examples = eval_data[tid][:20]
    preds = classify_by_likelihood(model, tokenizer, examples, tid, device)
    refs = [ex['response'] for ex in examples]
    correct = sum(1 for p, r in zip(preds, refs) if p.lower() == r.lower())
    print(f'  {tid} log-likelihood accuracy: {correct}/{len(preds)} = {correct/len(preds)*100:.0f}%')
    for i in range(min(5, len(preds))):
        print(f'    [{i}] ref={refs[i]:10s} pred={preds[i]}')
