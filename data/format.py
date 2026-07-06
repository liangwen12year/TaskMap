"""
Format each dataset into a unified instruction-response template.

Rules from the paper (Section 4.4):
- Classification labels use natural-language strings, not numeric IDs
- Math targets include a final-answer delimiter; scoring extracts the final normalized answer
- Code targets are fenced Python functions; execution evaluates first decoded sample at temperature 0
- All templates differ between train and eval to prevent template memorization
- Task descriptions avoid dataset names (no "SST-2" or "GSM8K")
- Exact duplicate input-target pairs removed across train/validation/test
"""

from datasets import load_dataset
from data.config import KNOWN_TASKS, COLD_START_TASKS, MAX_EXAMPLES_PER_TASK
import hashlib


ANSWER_DELIMITER = "\n\nAnswer: "
CODE_FENCE_START = "```python\n"
CODE_FENCE_END = "\n```"

TRAIN_TEMPLATE = (
    "Task: {description}\n\n"
    "Input: {input}\n\n"
    "Response:{response}"
)

EVAL_TEMPLATE = (
    "Instruction: {description}\n\n"
    "{input}\n\n"
    "Output:{response}"
)


# ── Per-task formatting functions ──

def format_sst2(example, split="train"):
    label_map = {0: "negative", 1: "positive"}
    return {
        "input": example["sentence"],
        "response": label_map.get(example["label"], "unknown"),
    }


def format_agnews(example, split="train"):
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}
    return {
        "input": example["text"],
        "response": label_map.get(example["label"], "unknown"),
    }


def format_boolq(example, split="train"):
    return {
        "input": f"Passage: {example['passage']}\nQuestion: {example['question']}",
        "response": "yes" if example["answer"] else "no",
    }


def format_squad(example, split="train"):
    answers = example["answers"]["text"]
    answer = answers[0] if answers else ""
    return {
        "input": f"Context: {example['context']}\nQuestion: {example['question']}",
        "response": answer,
    }


def format_xsum(example, split="train"):
    return {
        "input": example["document"],
        "response": example["summary"],
    }


def format_samsum(example, split="train"):
    return {
        "input": example["dialogue"],
        "response": example["summary"],
    }


def format_wmt14_ende(example, split="train"):
    trans = example["translation"]
    return {
        "input": trans["en"],
        "response": trans["de"],
    }


def format_wmt16_enro(example, split="train"):
    trans = example["translation"]
    return {
        "input": trans["en"],
        "response": trans["ro"],
    }


def format_gsm8k(example, split="train"):
    answer_text = example["answer"]
    return {
        "input": example["question"],
        "response": answer_text,
    }


def format_svamp(example, split="train"):
    return {
        "input": example["Body"] + " " + example["Question"],
        "response": str(example["Answer"]),
    }


def format_apps_intro(example, split="train"):
    return {
        "input": example["question"],
        "response": example.get("solutions", ""),
    }


def format_mbpp(example, split="train"):
    return {
        "input": example["text"],
        "response": example["code"],
    }


# ── Cold-start formatters ──

def format_trec6(example, split="train"):
    label_map = {0: "Abbreviation", 1: "Entity", 2: "Description",
                 3: "Human", 4: "Location", 5: "Number"}
    return {
        "input": example["text"],
        "response": label_map.get(example["coarse_label"], "unknown"),
    }


def format_multirc(example, split="train"):
    return {
        "input": f"Passage: {example['paragraph']}\nQuestion: {example['question']}\nAnswer: {example['answer']}",
        "response": "yes" if example["label"] else "no",
    }


def format_cnn_dailymail(example, split="train"):
    return {
        "input": example["article"],
        "response": example["highlights"],
    }


def format_flores_enfr(example, split="train"):
    return {
        "input": example["sentence_eng_Latn"],
        "response": example["sentence_fra_Latn"],
    }


def format_asdiv(example, split="train"):
    return {
        "input": example["body"] + " " + example["question"],
        "response": example["answer"],
    }


def format_humaneval(example, split="train"):
    return {
        "input": example["prompt"],
        "response": example["canonical_solution"],
    }


FORMAT_FNS = {
    "sst2": format_sst2,
    "agnews": format_agnews,
    "boolq": format_boolq,
    "squad": format_squad,
    "xsum": format_xsum,
    "samsum": format_samsum,
    "wmt14_ende": format_wmt14_ende,
    "wmt16_enro": format_wmt16_enro,
    "gsm8k": format_gsm8k,
    "svamp": format_svamp,
    "apps_intro": format_apps_intro,
    "mbpp": format_mbpp,
    "trec6": format_trec6,
    "multirc": format_multirc,
    "cnn_dailymail": format_cnn_dailymail,
    "flores_enfr": format_flores_enfr,
    "asdiv": format_asdiv,
    "humaneval": format_humaneval,
}


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def format_dataset(task_id: str, dataset, split: str, description_idx: int = 0):
    """
    Format a dataset split into list of dicts with keys:
    task_id, family, description, instruction, response, full_text
    """
    all_tasks = {**KNOWN_TASKS, **COLD_START_TASKS}
    meta = all_tasks[task_id]
    fmt_fn = FORMAT_FNS[task_id]
    description = meta["descriptions"][description_idx]
    template = TRAIN_TEMPLATE if split == "train" else EVAL_TEMPLATE

    formatted = []
    seen_hashes = set()

    for example in dataset:
        try:
            result = fmt_fn(example, split)
        except (KeyError, TypeError, IndexError):
            continue

        inp = result["input"]
        resp = result["response"]
        if not inp or not resp:
            continue

        dedup_key = _content_hash(f"{inp}|||{resp}")
        if dedup_key in seen_hashes:
            continue
        seen_hashes.add(dedup_key)

        full_text = template.format(
            description=description,
            input=inp,
            response=" " + resp,
        )

        formatted.append({
            "task_id": task_id,
            "family": meta["family"],
            "description": description,
            "input": inp,
            "response": resp,
            "full_text": full_text,
        })

        if split == "train" and len(formatted) >= MAX_EXAMPLES_PER_TASK:
            break

    return formatted


def format_all_tasks(datasets: dict, split: str = "train", description_idx: int = 0):
    """Format all tasks from downloaded datasets into a unified list."""
    all_tasks = {**KNOWN_TASKS, **COLD_START_TASKS}
    all_formatted = {}

    for task_id, ds in datasets.items():
        meta = all_tasks[task_id]
        split_name = meta["split_map"].get(split)
        if split_name is None or split_name not in ds:
            print(f"Skipping {task_id}/{split}: split '{split_name}' not found")
            continue

        formatted = format_dataset(task_id, ds[split_name], split, description_idx)
        all_formatted[task_id] = formatted
        print(f"  {task_id}/{split}: {len(formatted)} examples (deduped)")

    return all_formatted
