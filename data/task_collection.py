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

# Curated task selection from VERIFIED task names in the SNI dataset.
# 41 training + 20 held-out (all names confirmed to exist in the dataset).
TRAIN_TASKS_SNI = [
    # QA / Reading comprehension
    "task001_quoref_question_generation",
    "task002_quoref_answer_generation",
    "task073_commonsenseqa_answer_generation",
    "task074_squad1.1_question_generation",
    "task075_squad1.1_answer_generation",
    "task080_piqa_answer_generation",
    "task082_babi_t1_single_supporting_fact_question_generation",
    "task083_babi_t1_single_supporting_fact_answer_generation",
    "task043_essential_terms_answering_incomplete_questions",
    "task044_essential_terms_identifying_essential_words",
    "task047_miscellaneous_answering_science_questions",
    "task061_ropes_answer_generation",
    # Classification / NLI
    "task022_cosmosqa_passage_inappropriate_binary",
    "task065_timetravel_consistent_sentence_classification",
    "task066_timetravel_binary_consistency_classification",
    "task069_abductivenli_classification",
    "task070_abductivenli_incorrect_classification",
    "task092_check_prime_classification",
    "task108_contextualabusedetection_classification",
    "task109_smsspamcollection_spamsmsdetection",
    "task088_identify_typo_verification",
    "task089_swap_words_verification",
    # Generation / Summarization
    "task023_cosmosqa_question_generation",
    "task024_cosmosqa_answer_generation",
    "task025_cosmosqa_incorrect_answer_generation",
    "task026_drop_question_generation",
    "task027_drop_answer_type_generation",
    "task028_drop_answer_generation",
    "task059_ropes_story_generation",
    "task060_ropes_question_generation",
    "task067_abductivenli_answer_generation",
    "task068_abductivenli_incorrect_answer_generation",
    "task103_facts2story_long_text_generation",
    "task105_story_cloze-rocstories_sentence_generation",
    "task110_logic2text_sentence_generation",
    # Paraphrase / Simplification
    "task045_miscellaneous_sentence_paraphrasing",
    "task111_asset_sentence_simplification",
    # Reasoning / Math
    "task062_bigbench_repeat_copy_logic",
    "task063_first_i_elements",
    "task085_unnatural_addsub_arithmetic",
    # Misc
    "task046_miscellaneous_question_typing",
]

# Held-out tasks: 20 tasks from the TEST split (unseen during training)
HOLDOUT_TASKS_SNI = [
    "task020_mctaco_span_based_question",
    "task033_winogrande_answer_generation",
    "task034_winogrande_question_modification_object",
    "task036_qasc_topic_word_to_generate_related_fact",
    "task039_qasc_find_overlapping_words",
    "task050_multirc_answerability",
    "task102_commongen_sentence_generation",
    "task121_zest_text_modification",
    "task133_winowhy_reason_plausibility_detection",
    "task1152_bard_analogical_reasoning_causation",
    "task1342_amazon_us_reviews_title",
    "task1344_glue_entailment_classification",
    "task1345_glue_qqp_question_paraprashing",
    "task1356_xlsum_title_generation",
    "task1385_anli_r1_entailment",
    "task1386_anli_r2_entailment",
    "task1387_anli_r3_entailment",
    "task1388_cb_entailment",
    "task1394_meta_woz_task_classification",
    "task1409_dart_text_generation",
]


def load_sni_dataset(cache_dir=None):
    """Load the full SNI dataset (both train and test splits, concatenated)."""
    from datasets import concatenate_datasets
    print("Loading Super-NaturalInstructions dataset (both splits)...")
    all_splits = []
    for split in ["train", "test"]:
        try:
            ds = load_dataset("Muennighoff/natural-instructions", split=split,
                              cache_dir=cache_dir, verification_mode="no_checks")
            print(f"  {split}: {len(ds)} examples")
            all_splits.append(ds)
        except Exception as e:
            print(f"  {split}: failed ({e})")
    if not all_splits:
        raise RuntimeError("Could not load any split of natural-instructions")
    combined = concatenate_datasets(all_splits)
    print(f"  Combined: {len(combined)} examples")
    return combined


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
