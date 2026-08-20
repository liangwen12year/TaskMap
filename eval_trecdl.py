"""
TREC-DL 2019 passage reranking — cold-start evaluation.

Scores BM25 top-100 candidates via Yes/No log-likelihood, then evaluates
with nDCG@10, MRR@10, MAP@100.  No model is trained on MS MARCO or TREC data.

The candidates JSON and qrels JSON are committed under data/trecdl/.
This script is appended to training runs on the cluster.

Usage (standalone, frozen baseline):
  python eval_trecdl.py --backbone Qwen/Qwen2.5-1.5B

Called from training scripts after model is ready:
  eval_trecdl_with_model(model, tokenizer, device, output_dir)
"""

import os
import sys
import json
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

TASK_DESC = (
    "Given a web search query and a candidate passage, "
    "determine whether the passage contains information "
    "that helps answer the query."
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "trecdl")


def load_candidates(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "candidates_top100.json")
    with open(path) as f:
        return json.load(f)


def load_qrels(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "qrels.json")
    with open(path) as f:
        return json.load(f)


def build_prompt(query, passage):
    return (
        f"Task: {TASK_DESC}\n\n"
        f"Query: {query}\n\n"
        f"Passage: {passage}\n\n"
        f"Does the passage help answer the query?\n\n"
        f"Response:"
    )


@torch.no_grad()
def score_pairs(model, tokenizer, candidates, device="cuda"):
    """Score query-passage pairs via log P(Yes) - log P(No)."""
    model.eval()

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_tok, no_tok = yes_ids[0], no_ids[0]
    print(f"  Token IDs: Yes={yes_ids}, No={no_ids}")

    rankings = {}
    total = sum(len(v["passages"]) for v in candidates.values())
    done = 0

    for qid, data in candidates.items():
        scored = []
        for p in data["passages"]:
            prompt = build_prompt(data["query"], p["passage"])
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(device)
            logits = model(**inputs).logits[0, -1, :]
            lp = torch.log_softmax(logits, dim=-1)
            score = lp[yes_tok].item() - lp[no_tok].item()
            scored.append({"pid": p["pid"], "score": score})
            done += 1
        scored.sort(key=lambda x: x["score"], reverse=True)
        rankings[qid] = scored
        if done % 500 == 0 or done == total:
            print(f"  {done}/{total} pairs scored")

    return rankings


def evaluate_rankings(rankings, qrels):
    """Compute nDCG@10, MRR@10, MAP@100."""
    try:
        import pytrec_eval
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q pytrec-eval-terrier")
        import pytrec_eval

    # Convert qrels values to int
    qrels_int = {
        qid: {pid: int(rel) for pid, rel in docs.items()}
        for qid, docs in qrels.items()
    }

    # Binary qrels: grade >= 2 is relevant
    binary_qrels = {
        qid: {pid: (1 if rel >= 2 else 0) for pid, rel in docs.items()}
        for qid, docs in qrels_int.items()
    }

    run = {
        qid: {p["pid"]: p["score"] for p in passages}
        for qid, passages in rankings.items()
    }

    ndcg_ev = pytrec_eval.RelevanceEvaluator(qrels_int, {"ndcg_cut_10"})
    mrr_ev = pytrec_eval.RelevanceEvaluator(binary_qrels, {"recip_rank"})
    map_ev = pytrec_eval.RelevanceEvaluator(binary_qrels, {"map_cut_100"})

    ndcg_r = ndcg_ev.evaluate(run)
    mrr_r = mrr_ev.evaluate(run)
    map_r = map_ev.evaluate(run)

    ndcg_vals = [v["ndcg_cut_10"] for v in ndcg_r.values()]
    mrr_vals = [v["recip_rank"] for v in mrr_r.values()]
    map_vals = [v["map_cut_100"] for v in map_r.values()]

    return {
        "ndcg@10": round(float(np.mean(ndcg_vals)), 4),
        "mrr@10": round(float(np.mean(mrr_vals)), 4),
        "map@100": round(float(np.mean(map_vals)), 4),
        "num_queries": len(ndcg_vals),
        "per_query_ndcg": {
            qid: round(v["ndcg_cut_10"], 4)
            for qid, v in ndcg_r.items()
        },
    }


def eval_bm25_baseline(candidates, qrels):
    """Evaluate the original BM25 ranking."""
    bm25_rankings = {}
    for qid, data in candidates.items():
        bm25_rankings[qid] = [
            {"pid": p["pid"], "score": float(p.get("bm25_rank", 0))}
            for p in data["passages"]
        ]
        # BM25 rank: lower is better, so score = -rank for sorting
        for i, p in enumerate(bm25_rankings[qid]):
            p["score"] = -float(i + 1)
    return evaluate_rankings(bm25_rankings, qrels)


def eval_trecdl_with_model(model, tokenizer, device, output_dir,
                           method_name="method", seed=42,
                           candidates_path=None, qrels_path=None):
    """Main entry point for calling from training scripts."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TREC-DL 2019 Passage Reranking: {method_name} (seed {seed})")
    print(f"{'='*60}")

    candidates = load_candidates(candidates_path)
    qrels = load_qrels(qrels_path)
    total_pairs = sum(len(v["passages"]) for v in candidates.values())
    print(f"  {len(candidates)} queries, {total_pairs} pairs")

    # BM25 baseline
    bm25 = eval_bm25_baseline(candidates, qrels)
    print(f"\n  BM25 baseline:")
    print(f"    nDCG@10={bm25['ndcg@10']:.4f}  MRR@10={bm25['mrr@10']:.4f}"
          f"  MAP@100={bm25['map@100']:.4f}")

    # Neural reranking
    print(f"\n  Scoring with {method_name}...")
    rankings = score_pairs(model, tokenizer, candidates, device)
    results = evaluate_rankings(rankings, qrels)
    print(f"\n  {method_name}:")
    print(f"    nDCG@10={results['ndcg@10']:.4f}  MRR@10={results['mrr@10']:.4f}"
          f"  MAP@100={results['map@100']:.4f}")

    # Save
    out = {
        "method": method_name,
        "seed": seed,
        "bm25": bm25,
        method_name: results,
    }
    out_path = os.path.join(output_dir, f"trecdl_{method_name}_s{seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output_dir", default="outputs/trecdl")
    args = parser.parse_args()

    from models.backbone import load_backbone
    model, tokenizer = load_backbone(args.backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    eval_trecdl_with_model(
        model, tokenizer, device, args.output_dir,
        method_name="frozen", seed=42,
    )
