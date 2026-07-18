"""
Evaluation for all TaskMap-12 tasks + cold-start tasks.

Paper Section 5.1: Report each dataset's standard metric.
- Classification (SST-2, AG News, BoolQ): accuracy
- QA (SQuAD): F1 and exact match
- Summarization (XSum, SAMSum): ROUGE-L
- Translation (WMT14 En-De, WMT16 En-Ro): sacreBLEU and chrF
- Math (GSM8K, SVAMP): exact normalized answer
- Code (APPS, MBPP): pass@1

Unified summary: convert all to percentage, unweighted macro-average.
"""

import re
import torch
import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer
import sacrebleu


# ── Metric functions ──

def accuracy(predictions: list, references: list) -> dict:
    """
    Fuzzy accuracy: checks if the reference label appears anywhere in the
    prediction, handling verbose model outputs like "The answer is positive"
    when the reference is just "positive".
    """
    correct = 0
    for p, r in zip(predictions, references):
        p_lower = p.strip().lower()
        r_lower = r.strip().lower()
        if p_lower == r_lower:
            correct += 1
        elif r_lower in p_lower:
            correct += 1
        elif p_lower.startswith(r_lower):
            correct += 1
    return {"accuracy": correct / max(len(predictions), 1) * 100}


def f1_em(predictions: list, references: list) -> dict:
    """F1 and Exact Match for extractive QA."""
    f1_scores, em_scores = [], []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().lower().split()
        ref_tokens = ref.strip().lower().split()
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            f1_scores.append(0.0)
            em_scores.append(0.0)
            continue
        precision = len(common) / max(len(pred_tokens), 1)
        recall = len(common) / max(len(ref_tokens), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores.append(f1 * 100)
        em_scores.append(100.0 if pred.strip().lower() == ref.strip().lower() else 0.0)
    return {"f1": np.mean(f1_scores), "em": np.mean(em_scores)}


def rouge_l(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    return {"rouge_l": np.mean(scores) * 100}


def sacrebleu_chrf(predictions: list, references: list) -> dict:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return {"sacrebleu": bleu.score, "chrf": chrf.score}


def exact_answer(predictions: list, references: list) -> dict:
    """Extract final number from prediction and compare."""
    def extract_number(text):
        text = text.replace(",", "").replace("$", "").strip()
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        return text.strip()

    correct = 0
    for pred, ref in zip(predictions, references):
        pred_num = extract_number(pred)
        ref_num = extract_number(ref)
        if pred_num == ref_num:
            correct += 1
    return {"exact_answer": correct / max(len(predictions), 1) * 100}


def pass_at_1(predictions: list, references: list, test_cases: list = None) -> dict:
    """Pass@1 for code generation. Requires execution in sandbox."""
    # Placeholder: actual execution requires sandboxed environment
    # For now, return 0 and log a warning
    print("  WARNING: pass@1 requires sandboxed execution. Returning 0.")
    return {"pass_at_1": 0.0}


METRIC_FNS = {
    "accuracy": accuracy,
    "f1_em": f1_em,
    "rouge_l": rouge_l,
    "sacrebleu_chrf": sacrebleu_chrf,
    "exact_answer": exact_answer,
    "pass_at_1": pass_at_1,
}

CLASSIFICATION_LABELS = {
    "sst2": ["positive", "negative"],
    "agnews": ["World", "Sports", "Business", "Science/Technology"],
    "boolq": ["yes", "no"],
    "multirc": ["yes", "no"],
    "trec6": ["abbreviation", "entity", "description", "human", "location", "number"],
}


@torch.no_grad()
def classify_by_likelihood(model, tokenizer, examples: list, task_id: str,
                           device: str = "cuda") -> list:
    """
    Classify by scoring each valid label's log-likelihood given the prompt.
    Returns the label with highest likelihood for each example.
    """
    labels = CLASSIFICATION_LABELS.get(task_id)
    if labels is None:
        return None

    model.eval()
    predictions = []

    label_token_ids = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        label_token_ids.append(ids)

    for ex in examples:
        prompt = ex["full_text"].split("Response:" if "Response:" in ex["full_text"] else "Output:")[0]
        prompt += "Response: "

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(device)
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        best_label = None
        best_score = float('-inf')
        for label, token_ids in zip(labels, label_token_ids):
            score = log_probs[token_ids[0]].item()
            if score > best_score:
                best_score = score
                best_label = label

        predictions.append(best_label)

    return predictions


# ── Generation ──

@torch.no_grad()
def generate_predictions(model, tokenizer, examples: list, max_new_tokens: int = 128,
                         device: str = "cuda"):
    """Generate predictions for a list of formatted examples."""
    model.eval()
    predictions = []
    for ex in examples:
        prompt = ex["full_text"].split("Response:" if "Response:" in ex["full_text"] else "Output:")[0]
        prompt += "Response:" if "Response:" in ex["full_text"] else "Output:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048 - max_new_tokens).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append(decoded)

    return predictions


def evaluate_task(model, tokenizer, task_id: str, examples: list, metric_name: str,
                  max_new_tokens: int = 128, device: str = "cuda") -> dict:
    """Evaluate a single task. Uses log-likelihood scoring for classification."""
    print(f"  Evaluating {task_id} ({len(examples)} examples, metric={metric_name})...")

    if metric_name == "accuracy" and task_id in CLASSIFICATION_LABELS:
        predictions = classify_by_likelihood(model, tokenizer, examples, task_id, device)
        if predictions is not None:
            references = [ex["response"] for ex in examples]
            correct = sum(1 for p, r in zip(predictions, references) if p.lower() == r.lower())
            scores = {"accuracy": correct / max(len(predictions), 1) * 100}
            return scores

    predictions = generate_predictions(model, tokenizer, examples, max_new_tokens, device)
    references = [ex["response"] for ex in examples]

    metric_fn = METRIC_FNS[metric_name]
    scores = metric_fn(predictions, references)
    return scores


def evaluate_all(model, tokenizer, eval_data: dict, task_configs: dict,
                 device: str = "cuda") -> dict:
    """
    Evaluate all tasks and compute macro-average.
    Returns per-task scores and the macro score.
    """
    all_scores = {}
    primary_scores = []

    for task_id, examples in eval_data.items():
        if task_id not in task_configs:
            continue
        meta = task_configs[task_id]
        scores = evaluate_task(
            model, tokenizer, task_id, examples,
            meta["metric"], meta["max_response_tokens"], device
        )
        all_scores[task_id] = scores
        primary = list(scores.values())[0]
        primary_scores.append(primary)
        print(f"    {task_id}: {scores}")

    macro = np.mean(primary_scores) if primary_scores else 0.0
    all_scores["macro_avg"] = macro
    print(f"\n  Macro average: {macro:.2f}")
    return all_scores


# ── Negative transfer (Section 5.2) ──

def compute_negative_transfer(multi_task_scores: dict, task_specific_scores: dict) -> dict:
    """
    I_t = Score(TaskSpecificLoRA_t) - Score(MultiTaskMethod_t)
    Positive I_t means the multi-task method hurts that task.
    """
    interference = {}
    for task_id in multi_task_scores:
        if task_id == "macro_avg" or task_id not in task_specific_scores:
            continue
        multi_score = list(multi_task_scores[task_id].values())[0]
        specific_score = list(task_specific_scores[task_id].values())[0]
        interference[task_id] = specific_score - multi_score

    i_values = list(interference.values())
    return {
        "per_task": interference,
        "mean_I_t": np.mean(i_values) if i_values else 0,
        "median_I_t": np.median(i_values) if i_values else 0,
        "worst_10pct": np.percentile(i_values, 90) if i_values else 0,
        "fraction_positive": sum(1 for v in i_values if v > 0) / max(len(i_values), 1),
    }
