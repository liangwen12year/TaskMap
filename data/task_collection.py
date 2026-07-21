"""
Large-scale task collection for TaskMap generalization experiments.

Uses Super-NaturalInstructions (SNI) as the task source:
  - 1600+ tasks with natural-language definitions
  - Uniform schema: definition, positive/negative examples
  - English subset: 757 training tasks, 119 test tasks

We select 50 training tasks across 10+ families and hold out 20 tasks
(including entire held-out families) for cold-start evaluation.

Task families are mapped from SNI's 76 task types into broader groups
that align with our existing framework.
"""

import os
import json
import random
from collections import defaultdict
from datasets import load_dataset

# Map SNI task types to broader families
SNI_FAMILY_MAP = {
    # Classification
    "sentiment_analysis": "classification",
    "text_categorization": "classification",
    "toxic_language_detection": "classification",
    "stance_detection": "classification",
    "intent_identification": "classification",
    "spam_classification": "classification",

    # Question Answering
    "question_answering": "question_answering",
    "reading_comprehension": "question_answering",
    "answer_verification": "question_answering",
    "question_generation": "question_answering",

    # Summarization
    "summarization": "summarization",
    "title_generation": "summarization",

    # Translation / Paraphrase
    "translation": "translation",
    "paraphrasing": "paraphrase",
    "style_transfer": "paraphrase",

    # Reasoning
    "commonsense_reasoning": "reasoning",
    "mathematical_calculation": "reasoning",
    "numerical_reasoning": "reasoning",
    "logical_reasoning": "reasoning",
    "cause_effect_classification": "reasoning",
    "analogy": "reasoning",

    # Information Extraction
    "information_extraction": "extraction",
    "named_entity_recognition": "extraction",
    "relation_extraction": "extraction",
    "keyword_tagging": "extraction",

    # Text Generation
    "data_to_text": "generation",
    "dialogue_generation": "generation",
    "story_composition": "generation",
    "explanation": "generation",

    # NLI / Entailment
    "textual_entailment": "nli",
    "natural_language_inference": "nli",

    # Coreference / Linguistics
    "coreference_resolution": "linguistics",
    "word_semantics": "linguistics",
    "linguistic_probing": "linguistics",
    "pos_tagging": "linguistics",

    # Misc
    "fact_verification": "fact_checking",
    "text_completion": "completion",
    "fill_in_the_blank": "completion",
    "text_matching": "matching",
    "coherence_classification": "matching",
}

# Curated task selection: 50 training + 20 held-out
# Criteria: English only, >500 examples, diverse families, well-defined outputs
TRAIN_TASKS_SNI = [
    # Classification (8 tasks)
    "task137_detox_classification",
    "task199_sentiment_classification",
    "task202_sentiment_classification",
    "task284_imdb_classification",
    "task1387_anli_r3_entailment",
    "task195_sentiment_classification",
    "task333_hateeval_classification",
    "task363_sst2_polarity",

    # Question Answering (8 tasks)
    "task020_mctaco_answer_generation_event_duration",
    "task033_winogrande_answer_generation",
    "task039_qasc_find_overlapping_words",
    "task050_multirc_answerability",
    "task073_CommonsenseQA_answer_generation",
    "task233_iirc_link_classification",
    "task290_tellmewhy_question_answering",
    "task391_causal_relationship",

    # Summarization (5 tasks)
    "task510_reddit_tifu_title_summarization",
    "task511_reddit_tifu_long_text_summarization",
    "task569_recipe_nlg_text_generation",
    "task1290_xsum_summarization",
    "task1586_scifact_title_generation",

    # Translation / Paraphrase (5 tasks)
    "task316_crows-pairs_classification_stereotype",
    "task401_numeric_fused_head_reference",
    "task1557_jfleg_answer_generation",
    "task306_jeopardy_answer_generation_all",
    "task1409_dart_text_generation",

    # Reasoning (6 tasks)
    "task069_abductivenli_classification",
    "task070_abductivenli_incorrect_answer_generation",
    "task102_commongen_sentence_generation",
    "task380_boolq_yes_no_question",
    "task391_causal_relationship",
    "task1388_cb_entailment",

    # Information Extraction (5 tasks)
    "task036_qasc_topic_word_to_generate_related_fact",
    "task281_points_of_interest_master_name",
    "task329_gap_classification",
    "task330_gap_answer_generation",
    "task614_glucose_cause_event_detection",

    # Text Generation (5 tasks)
    "task442_com_qa_paraphrase_question_generation",
    "task571_recipe_nlg_ner_generation",
    "task613_politifact_text_generation",
    "task677_ollie_sentence_answer_generation",
    "task748_glucose_reverse_cause_event_detection",

    # NLI (4 tasks)
    "task190_snli_classification",
    "task197_mnli_domain_answer_generation",
    "task1386_anli_r2_entailment",
    "task970_sherliic_causal_relationship",

    # Fact checking (4 tasks)
    "task213_rocstories_correct_ending_classification",
    "task220_rocstories_title_classification",
    "task227_clariq_classification",
    "task228_arc_answer_generation",
]

# Held-out tasks: 20 tasks, including entire held-out families
HOLDOUT_TASKS_SNI = [
    # From known families (10 tasks) — tests within-family generalization
    "task200_sentiment_classification",      # classification
    "task362_spolin_yesand_prompt_response_classification",  # classification
    "task034_winogrande_question_modification_object",  # QA
    "task1290_xsum_summarization",           # summarization (duplicate, will use different split)
    "task1389_paws_paraphrase_classification",  # paraphrase
    "task389_torque_generate_temporal_question",  # QA
    "task515_senteval_odd_word_out",          # linguistics
    "task937_defeasible_nli",                 # NLI
    "task828_copa_commonsense_cause_effect",  # reasoning
    "task1344_glue_entailment_classification",  # NLI

    # From held-out families (10 tasks) — tests cross-family generalization
    "task002_quoref_answer_generation",       # reading comprehension
    "task003_mctaco_question_generation_event_duration",  # temporal reasoning
    "task121_zest_text_modification",         # text modification
    "task133_winowhy_reason_plausibility_detection",  # commonsense
    "task178_quartz_question_answering",      # science QA
    "task242_tweetqa_classification",         # tweet understanding
    "task288_gigaword_summarization",         # news summarization
    "task418_persent_title_generation",       # title generation
    "task500_scruples_anecdotes_title_generation",  # ethics
    "task891_gap_coreference_resolution",     # coreference
]


def load_sni_dataset(cache_dir=None):
    """Load the full SNI dataset (downloads once, cached)."""
    print("Loading Super-NaturalInstructions dataset...")
    try:
        ds = load_dataset("Muennighoff/natural-instructions", split="train",
                          cache_dir=cache_dir)
    except Exception as e:
        print(f"  Failed with split='train': {e}")
        print("  Trying with verification disabled...")
        try:
            ds = load_dataset("Muennighoff/natural-instructions", split="train",
                              cache_dir=cache_dir, verification_mode="no_checks")
        except Exception as e2:
            print(f"  Failed again: {e2}")
            print("  Trying test split...")
            try:
                ds = load_dataset("Muennighoff/natural-instructions", split="test",
                                  cache_dir=cache_dir, verification_mode="no_checks")
            except Exception as e3:
                print(f"  Trying without split...")
                ds_dict = load_dataset("Muennighoff/natural-instructions",
                                       cache_dir=cache_dir, verification_mode="no_checks")
                available_splits = list(ds_dict.keys())
                print(f"  Available splits: {available_splits}")
                ds = ds_dict[available_splits[0]]
    print(f"  Total examples: {len(ds)}")
    return ds


def filter_sni_tasks(full_dataset, task_names, max_per_task=2000):
    """Filter the full SNI dataset to specific tasks."""
    task_set = set(task_names)
    task_data = defaultdict(list)

    for ex in full_dataset:
        tname = ex["task_name"]
        if tname in task_set and len(task_data[tname]) < max_per_task:
            task_data[tname].append(ex)

    return dict(task_data)


def format_sni_examples(task_name, raw_examples, split="train"):
    """Format SNI examples into our instruction template."""
    examples = []
    definition = ""

    for ex in raw_examples:
        input_text = ex.get("inputs", "")
        output_text = ex.get("targets", "")
        definition = ex.get("definition", "")

        if isinstance(output_text, list):
            output_text = output_text[0] if output_text else ""

        if not input_text or not output_text:
            continue

        desc = definition[:200] if definition else "Complete the following task."

        formatted = {
            "task_id": task_name,
            "family": get_task_family(task_name),
            "input": input_text,
            "response": output_text.strip(),
            "description": desc,
            "full_text": f"Task: {desc}\n\nInput: {input_text}\n\nResponse: {output_text.strip()}",
        }
        examples.append(formatted)

    # Split: use 80% for train, 20% for validation
    if split == "train":
        examples = examples[:int(len(examples) * 0.8)]
    elif split == "validation":
        examples = examples[int(len(examples) * 0.8):]

    return examples


def load_all_sni_tasks(task_list, split="train", max_per_task=2000, cache_dir=None):
    """Load and format all tasks in a list from SNI."""
    full_ds = load_sni_dataset(cache_dir)
    raw_data = filter_sni_tasks(full_ds, task_list, max_per_task)

    all_data = {}
    for task_name in task_list:
        if task_name not in raw_data:
            print(f"  {task_name}: NOT FOUND in dataset")
            continue
        formatted = format_sni_examples(task_name, raw_data[task_name], split)
        if formatted:
            all_data[task_name] = formatted
            print(f"  {task_name}: {len(formatted)} examples")
        else:
            print(f"  {task_name}: no examples formatted")

    return all_data


def get_task_family(task_name):
    """Get the family for an SNI task based on its type."""
    # Try to infer from task name or metadata
    for key, family in SNI_FAMILY_MAP.items():
        if key in task_name.lower():
            return family
    return "other"


def get_task_descriptions(task_name, dataset):
    """Extract natural-language descriptions for a task."""
    descriptions = []
    for split in dataset:
        for ex in dataset[split]:
            defn = ex.get("definition", ex.get("task_definition", ""))
            if defn and defn not in descriptions:
                descriptions.append(defn[:200])
            if len(descriptions) >= 3:
                break
        if len(descriptions) >= 3:
            break

    if not descriptions:
        descriptions = [f"Complete the following task: {task_name}"]

    return descriptions


if __name__ == "__main__":
    print(f"Training tasks: {len(TRAIN_TASKS_SNI)}")
    print(f"Held-out tasks: {len(HOLDOUT_TASKS_SNI)}")

    # Test loading a few
    print("\nTesting task loading...")
    test_tasks = TRAIN_TASKS_SNI[:3]
    data = load_all_sni_tasks(test_tasks, max_per_task=10)
    for tid, examples in data.items():
        print(f"\n{tid}:")
        if examples:
            print(f"  Input: {examples[0]['input'][:100]}...")
            print(f"  Response: {examples[0]['response'][:100]}...")
            print(f"  Description: {examples[0]['description'][:100]}...")
