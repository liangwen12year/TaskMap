"""
Task metadata for TaskMap-12 multi-task mixture + 6 cold-start tasks.

Each task has:
- task_id: internal identifier (no dataset names in descriptions)
- family: one of 6 families for topology loss
- description: 3 paraphrases for robustness evaluation
- dataset: HuggingFace dataset name for downloading
- split_map: mapping to train/validation/test splits
- max_response_tokens: truncation length for the response
- metric: primary evaluation metric
- format_fn_name: key into FORMAT_FNS (defined in format.py)
"""

TASK_FAMILIES = [
    "classification",
    "question_answering",
    "summarization",
    "translation",
    "mathematical_reasoning",
    "code_generation",
]

KNOWN_TASKS = {
    # ── Classification ──
    "sst2": {
        "family": "classification",
        "descriptions": [
            "Classify the text into the requested category.",
            "Determine the sentiment expressed in the given text.",
            "Read the text and assign it to the appropriate sentiment class.",
        ],
        "dataset": "stanfordnlp/sst2",
        "subset": None,
        "split_map": {"train": "train", "validation": "validation", "test": "validation"},
        "max_response_tokens": 16,
        "metric": "accuracy",
    },
    "agnews": {
        "family": "classification",
        "descriptions": [
            "Classify the text into the requested category.",
            "Categorize this news article into its topic.",
            "Read the article and determine which category it belongs to.",
        ],
        "dataset": "fancyzhx/ag_news",
        "subset": None,
        "split_map": {"train": "train", "validation": "test", "test": "test"},
        "max_response_tokens": 16,
        "metric": "accuracy",
    },

    # ── Question Answering ──
    "boolq": {
        "family": "question_answering",
        "descriptions": [
            "Answer a question using the provided information.",
            "Read the passage and answer the yes/no question.",
            "Determine whether the statement is true or false based on the passage.",
        ],
        "dataset": "google/boolq",
        "subset": None,
        "split_map": {"train": "train", "validation": "validation", "test": "validation"},
        "max_response_tokens": 16,
        "metric": "accuracy",
    },
    "squad": {
        "family": "question_answering",
        "descriptions": [
            "Answer a question using the provided information.",
            "Extract the answer to the question from the given context.",
            "Read the passage and provide the answer to the question.",
        ],
        "dataset": "rajpurkar/squad",
        "subset": None,
        "split_map": {"train": "train", "validation": "validation", "test": "validation"},
        "max_response_tokens": 96,
        "metric": "f1_em",
    },

    # ── Summarization ──
    "xsum": {
        "family": "summarization",
        "descriptions": [
            "Write a faithful, concise summary of the source.",
            "Summarize the following document in one sentence.",
            "Produce a brief summary capturing the main point of the text.",
        ],
        "dataset": "EdinburghNLP/xsum",
        "subset": None,
        "split_map": {"train": "train", "validation": "validation", "test": "test"},
        "max_response_tokens": 192,
        "metric": "rouge_l",
    },
    "samsum": {
        "family": "summarization",
        "descriptions": [
            "Write a faithful, concise summary of the source.",
            "Summarize the dialogue into a brief description.",
            "Provide a short summary of the conversation.",
        ],
        "dataset": "Samsung/samsum",
        "subset": None,
        "split_map": {"train": "train", "validation": "validation", "test": "test"},
        "max_response_tokens": 192,
        "metric": "rouge_l",
    },

    # ── Translation ──
    "wmt14_ende": {
        "family": "translation",
        "descriptions": [
            "Translate the source sentence into the target language.",
            "Convert the following English text into German.",
            "Provide an accurate German translation of the English sentence.",
        ],
        "dataset": "wmt14",
        "subset": "de-en",
        "split_map": {"train": "train", "validation": "validation", "test": "test"},
        "max_response_tokens": 128,
        "metric": "sacrebleu_chrf",
    },
    "wmt16_enro": {
        "family": "translation",
        "descriptions": [
            "Translate the source sentence into the target language.",
            "Convert the following English text into Romanian.",
            "Provide an accurate Romanian translation of the English sentence.",
        ],
        "dataset": "wmt16",
        "subset": "ro-en",
        "split_map": {"train": "train", "validation": "validation", "test": "test"},
        "max_response_tokens": 128,
        "metric": "sacrebleu_chrf",
    },

    # ── Mathematical Reasoning ──
    "gsm8k": {
        "family": "mathematical_reasoning",
        "descriptions": [
            "Solve the mathematical word problem and give the final answer.",
            "Work through the math problem step by step and provide the answer.",
            "Read the word problem, compute the solution, and state the final number.",
        ],
        "dataset": "openai/gsm8k",
        "subset": "main",
        "split_map": {"train": "train", "validation": "test", "test": "test"},
        "max_response_tokens": 128,
        "metric": "exact_answer",
    },
    "svamp": {
        "family": "mathematical_reasoning",
        "descriptions": [
            "Solve the mathematical word problem and give the final answer.",
            "Work through the arithmetic problem and provide the result.",
            "Compute the answer to the given math question.",
        ],
        "dataset": "ChilleD/SVAMP",
        "subset": None,
        "split_map": {"train": "train", "validation": "test", "test": "test"},
        "max_response_tokens": 128,
        "metric": "exact_answer",
    },

    # ── Code Generation ──
    "apps_intro": {
        "family": "code_generation",
        "descriptions": [
            "Write a correct Python function satisfying the specification.",
            "Implement the described function in Python.",
            "Generate Python code that solves the given programming problem.",
        ],
        "dataset": "codeparrot/apps",
        "subset": "introductory",
        "split_map": {"train": "train", "validation": "test", "test": "test"},
        "max_response_tokens": 384,
        "metric": "pass_at_1",
    },
    "mbpp": {
        "family": "code_generation",
        "descriptions": [
            "Write a correct Python function satisfying the specification.",
            "Implement the described Python function.",
            "Generate the Python code for the given task description.",
        ],
        "dataset": "google-research-datasets/mbpp",
        "subset": "full",
        "split_map": {"train": "train", "validation": "validation", "test": "test"},
        "max_response_tokens": 384,
        "metric": "pass_at_1",
    },
}

COLD_START_TASKS = {
    "trec6": {
        "family": "classification",
        "descriptions": [
            "Classify the question into its type category.",
            "Determine the category of the given question.",
            "Identify what type of answer the question is seeking.",
        ],
        "dataset": "CogComp/trec",
        "subset": None,
        "split_map": {"test": "test"},
        "max_response_tokens": 16,
        "metric": "accuracy",
    },
    "multirc": {
        "family": "question_answering",
        "descriptions": [
            "Answer the question based on the provided passage.",
            "Read the text and determine the answer to the question.",
            "Use the passage to answer the following question.",
        ],
        "dataset": "aps/super_glue",
        "subset": "multirc",
        "split_map": {"test": "validation"},
        "max_response_tokens": 16,
        "metric": "accuracy",
    },
    "cnn_dailymail": {
        "family": "summarization",
        "descriptions": [
            "Write a faithful, concise summary of the source.",
            "Summarize the following news article.",
            "Produce a brief summary of the article.",
        ],
        "dataset": "abisee/cnn_dailymail",
        "subset": "3.0.0",
        "split_map": {"test": "test"},
        "max_response_tokens": 192,
        "metric": "rouge_l",
    },
    "flores_enfr": {
        "family": "translation",
        "descriptions": [
            "Translate the source sentence into the target language.",
            "Convert the following English text into French.",
            "Provide an accurate French translation of the English sentence.",
        ],
        "dataset": "facebook/flores",
        "subset": "eng_Latn-fra_Latn",
        "split_map": {"test": "devtest"},
        "max_response_tokens": 128,
        "metric": "sacrebleu_chrf",
    },
    "asdiv": {
        "family": "mathematical_reasoning",
        "descriptions": [
            "Solve the mathematical word problem and give the final answer.",
            "Work through the math problem and provide the answer.",
            "Compute the solution to the given word problem.",
        ],
        "dataset": "EleutherAI/asdiv",
        "subset": None,
        "split_map": {"test": "test"},
        "max_response_tokens": 128,
        "metric": "exact_answer",
    },
    "humaneval": {
        "family": "code_generation",
        "descriptions": [
            "Write a correct Python function satisfying the specification.",
            "Complete the given Python function.",
            "Implement the function as described in the docstring.",
        ],
        "dataset": "openai/openai_humaneval",
        "subset": None,
        "split_map": {"test": "test"},
        "max_response_tokens": 384,
        "metric": "pass_at_1",
    },
}

MAX_EXAMPLES_PER_TASK = 40_000
OVERSAMPLING_CAP = 2
FAMILY_TO_TASKS = {}
for task_id, meta in KNOWN_TASKS.items():
    fam = meta["family"]
    FAMILY_TO_TASKS.setdefault(fam, []).append(task_id)

FAMILY_PAIRS = []
for fam, tasks in FAMILY_TO_TASKS.items():
    for i, t1 in enumerate(tasks):
        for t2 in tasks[i + 1:]:
            FAMILY_PAIRS.append((t1, t2))
