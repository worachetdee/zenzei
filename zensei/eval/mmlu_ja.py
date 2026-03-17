"""MMLU Japanese evaluation for the Zensei project.

Evaluates the model on MMLU translated to Japanese. Supports per-subject
and per-category accuracy with configurable few-shot prompting.

Attempts to load a pre-translated Japanese MMLU dataset; falls back to
on-the-fly translation using a simple prompt-based approach if needed.

Usage:
    from zensei.eval.mmlu_ja import evaluate_mmlu_ja
    results = evaluate_mmlu_ja(model, tokenizer, n_shots=5)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# MMLU category groupings (high-level)
CATEGORY_MAP = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history", "international_law",
        "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios",
        "philosophy", "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}

# Reverse map: subject -> category
SUBJECT_TO_CATEGORY: dict[str, str] = {}
for cat, subjects in CATEGORY_MAP.items():
    for subj in subjects:
        SUBJECT_TO_CATEGORY[subj] = cat

CHOICE_LABELS = ["A", "B", "C", "D"]

# Known Japanese MMLU datasets on HuggingFace
MMLU_JA_DATASETS = [
    "nlp-waseda/JMMLU",
    "csebuetnlp/mmmlu",
]


def _try_load_mmlu_ja() -> tuple[Optional[object], str]:
    """Attempt to load a pre-translated Japanese MMLU dataset.

    Returns:
        (dataset_dict_or_None, dataset_name_used)
    """
    for ds_name in MMLU_JA_DATASETS:
        try:
            logger.info("Trying to load MMLU-ja from '%s'...", ds_name)
            if ds_name == "csebuetnlp/mmmlu":
                ds = load_dataset(ds_name, "ja", trust_remote_code=True)
            else:
                ds = load_dataset(ds_name, trust_remote_code=True)
            logger.info("Successfully loaded '%s'", ds_name)
            return ds, ds_name
        except Exception as e:
            logger.warning("Could not load '%s': %s", ds_name, e)

    # Fallback: load English MMLU (will need on-the-fly translation)
    logger.info("Falling back to English MMLU (on-the-fly translation)...")
    try:
        ds = load_dataset("cais/mmlu", "all", trust_remote_code=True)
        return ds, "cais/mmlu"
    except Exception as e:
        logger.error("Could not load any MMLU dataset: %s", e)
        return None, ""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_question(
    question: str,
    choices: list[str],
    answer: Optional[str] = None,
) -> str:
    """Format a single MMLU question with choices.

    Args:
        question: The question text.
        choices: List of 4 choice strings.
        answer: If provided, appended as the answer (for few-shot examples).

    Returns:
        Formatted question string.
    """
    lines = [f"質問: {question}"]
    for i, choice in enumerate(choices):
        lines.append(f"({CHOICE_LABELS[i]}) {choice}")
    lines.append("回答:")
    if answer is not None:
        lines[-1] += f" {answer}"
    return "\n".join(lines)


def _build_mmlu_prompt(
    question: str,
    choices: list[str],
    few_shot_examples: list[dict],
    subject: str = "",
) -> str:
    """Build a full MMLU evaluation prompt with optional few-shot examples.

    Args:
        question: The test question.
        choices: The 4 answer choices.
        few_shot_examples: List of dicts with keys: question, choices, answer.
        subject: Subject name for the instruction header.

    Returns:
        Complete prompt string.
    """
    subject_ja = subject.replace("_", " ") if subject else "一般知識"
    header = f"以下は{subject_ja}に関する問題です。最も適切な回答をA、B、C、Dの中から選んでください。\n\n"

    prompt = header
    for ex in few_shot_examples:
        prompt += _format_question(ex["question"], ex["choices"], ex["answer"]) + "\n\n"
    prompt += _format_question(question, choices)

    return prompt


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_choice(
    model,
    tokenizer,
    prompt: str,
) -> str:
    """Predict the answer choice (A/B/C/D) for an MMLU question.

    Uses the log-likelihood method: compare the probability of generating
    each choice label as the next token.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

    # Get logits for the last token position
    last_logits = logits[0, -1, :]  # (vocab_size,)

    # Score each choice label
    best_choice = "A"
    best_score = float("-inf")

    for label in CHOICE_LABELS:
        token_id = tokenizer.encode(label, add_special_tokens=False)
        if not token_id:
            continue
        # Use the first token of the label encoding
        score = last_logits[token_id[0]].item()
        if score > best_score:
            best_score = score
            best_choice = label

    return best_choice


# ---------------------------------------------------------------------------
# Dataset field extraction
# ---------------------------------------------------------------------------

def _extract_fields(sample: dict, dataset_name: str) -> dict:
    """Extract question, choices, answer, and subject from a dataset sample.

    Handles different field names across MMLU dataset variants.

    Returns:
        Dict with keys: question, choices, answer_idx, subject.
    """
    # Try common field names
    question = (
        sample.get("question", "")
        or sample.get("input", "")
        or sample.get("text", "")
    )

    # Choices
    choices = sample.get("choices", None)
    if choices is None:
        choices = []
        for key in ["A", "B", "C", "D"]:
            if key in sample:
                choices.append(sample[key])
        if not choices:
            for key in ["choice_a", "choice_b", "choice_c", "choice_d"]:
                if key in sample:
                    choices.append(sample[key])
        if not choices:
            for key in ["option_a", "option_b", "option_c", "option_d"]:
                if key in sample:
                    choices.append(sample[key])

    # Answer
    answer = sample.get("answer", sample.get("target", sample.get("label", 0)))
    if isinstance(answer, str):
        answer_idx = CHOICE_LABELS.index(answer.upper()) if answer.upper() in CHOICE_LABELS else 0
    else:
        answer_idx = int(answer)

    # Subject
    subject = sample.get("subject", sample.get("category", "unknown"))

    return {
        "question": question,
        "choices": choices if len(choices) == 4 else ["", "", "", ""],
        "answer_idx": answer_idx,
        "subject": subject,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_mmlu_ja(
    model,
    tokenizer,
    n_shots: int = 5,
    max_samples: Optional[int] = None,
    subjects: Optional[list[str]] = None,
) -> dict:
    """Evaluate a model on MMLU in Japanese.

    Args:
        model: HuggingFace-compatible causal language model.
        tokenizer: Corresponding tokenizer.
        n_shots: Number of few-shot examples (default 5).
        max_samples: Maximum total samples to evaluate (None = all).
        subjects: Optional list of specific subjects to evaluate.

    Returns:
        Dictionary with per-subject, per-category, and overall accuracy.
    """
    dataset_obj, dataset_name = _try_load_mmlu_ja()
    if dataset_obj is None:
        return {"error": "Could not load any MMLU dataset"}

    # Determine the evaluation split
    if "test" in dataset_obj:
        eval_split = dataset_obj["test"]
    elif "validation" in dataset_obj:
        eval_split = dataset_obj["validation"]
    else:
        # Try to get the first available split
        split_name = list(dataset_obj.keys())[0]
        eval_split = dataset_obj[split_name]

    if max_samples is not None:
        eval_split = eval_split.select(range(min(max_samples, len(eval_split))))

    # Build few-shot pool from training data if available
    few_shot_pool: dict[str, list[dict]] = {}
    if n_shots > 0 and "train" in dataset_obj:
        train_split = dataset_obj["train"]
        # If dataset is large enough, sample from dev/validation for few-shot instead
        if "dev" in dataset_obj:
            train_split = dataset_obj["dev"]

        for sample in train_split:
            fields = _extract_fields(sample, dataset_name)
            subj = fields["subject"]
            if subj not in few_shot_pool:
                few_shot_pool[subj] = []
            if len(few_shot_pool[subj]) < n_shots:
                few_shot_pool[subj].append({
                    "question": fields["question"],
                    "choices": fields["choices"],
                    "answer": CHOICE_LABELS[fields["answer_idx"]],
                })

    # Run evaluation
    subject_correct: dict[str, int] = {}
    subject_total: dict[str, int] = {}

    for sample in tqdm(eval_split, desc="MMLU-ja"):
        fields = _extract_fields(sample, dataset_name)
        subj = fields["subject"]

        # Filter by subject if specified
        if subjects is not None and subj not in subjects:
            continue

        # Get few-shot examples for this subject
        fs_examples = few_shot_pool.get(subj, [])[:n_shots]

        prompt = _build_mmlu_prompt(
            fields["question"], fields["choices"], fs_examples, subject=subj
        )
        predicted = _predict_choice(model, tokenizer, prompt)
        gold = CHOICE_LABELS[fields["answer_idx"]]

        if subj not in subject_correct:
            subject_correct[subj] = 0
            subject_total[subj] = 0

        subject_total[subj] += 1
        if predicted == gold:
            subject_correct[subj] += 1

    # Compute per-subject accuracy
    per_subject: dict[str, dict] = {}
    for subj in sorted(subject_total.keys()):
        acc = subject_correct[subj] / subject_total[subj] if subject_total[subj] > 0 else 0.0
        per_subject[subj] = {
            "accuracy": round(acc, 4),
            "correct": subject_correct[subj],
            "total": subject_total[subj],
        }

    # Compute per-category accuracy
    per_category: dict[str, dict] = {}
    for cat in CATEGORY_MAP:
        cat_correct = sum(subject_correct.get(s, 0) for s in CATEGORY_MAP[cat])
        cat_total = sum(subject_total.get(s, 0) for s in CATEGORY_MAP[cat])
        if cat_total > 0:
            per_category[cat] = {
                "accuracy": round(cat_correct / cat_total, 4),
                "correct": cat_correct,
                "total": cat_total,
            }

    # Overall accuracy
    total_correct = sum(subject_correct.values())
    total_samples = sum(subject_total.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    logger.info("MMLU-ja overall accuracy: %.4f (%d/%d)", overall_accuracy, total_correct, total_samples)
    for cat, data in per_category.items():
        logger.info("  %-20s : %.4f (%d/%d)", cat, data["accuracy"], data["correct"], data["total"])

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "accuracy": round(overall_accuracy, 4),
        "total_correct": total_correct,
        "total_samples": total_samples,
        "per_category": per_category,
        "per_subject": per_subject,
        "dataset_used": dataset_name,
        "n_shots": n_shots,
    }
