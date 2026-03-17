"""JGLUE benchmark evaluation for the Zensei project.

Supports tasks:
  - JNLI (natural language inference)
  - JSQuAD (question answering)
  - JCommonsenseQA (commonsense multiple choice)
  - MARC-ja (sentiment classification)

Reference: https://github.com/yahoojapan/JGLUE
Datasets: shunk031/JGLUE on HuggingFace

Usage:
    from zensei.eval.jglue import evaluate_jglue
    results = evaluate_jglue(model, tokenizer)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

JGLUE_TASKS = ["jnli", "jsquad", "jcommonsenseqa", "marc_ja"]

JNLI_LABEL_MAP = {0: "entailment", 1: "contradiction", 2: "neutral"}
MARC_LABEL_MAP = {0: "negative", 1: "positive"}


# ---------------------------------------------------------------------------
# Text generation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate text from a prompt using greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the generated portion
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Japanese text normalization
# ---------------------------------------------------------------------------

def _normalize_ja(text: str) -> str:
    """Normalize Japanese text for comparison."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    # Remove extra whitespace
    text = re.sub(r"\s+", "", text)
    return text


# ---------------------------------------------------------------------------
# Character-level F1 for Japanese QA
# ---------------------------------------------------------------------------

def _char_f1(prediction: str, reference: str) -> float:
    """Compute character-level F1 between prediction and reference."""
    pred_chars = list(_normalize_ja(prediction))
    ref_chars = list(_normalize_ja(reference))

    if not ref_chars and not pred_chars:
        return 1.0
    if not ref_chars or not pred_chars:
        return 0.0

    common = sum(1 for c in pred_chars if c in ref_chars)
    if common == 0:
        return 0.0

    precision = common / len(pred_chars)
    recall = common / len(ref_chars)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> float:
    """Compute exact match after normalization."""
    return 1.0 if _normalize_ja(prediction) == _normalize_ja(reference) else 0.0


# ---------------------------------------------------------------------------
# Few-shot prompt builders
# ---------------------------------------------------------------------------

def _build_jnli_prompt(premise: str, hypothesis: str, few_shot_examples: list[dict]) -> str:
    """Build a JNLI evaluation prompt."""
    prompt = ""
    for ex in few_shot_examples:
        prompt += (
            f"前提: {ex['premise']}\n"
            f"仮説: {ex['hypothesis']}\n"
            f"関係: {ex['label']}\n\n"
        )
    prompt += (
        f"前提: {premise}\n"
        f"仮説: {hypothesis}\n"
        f"関係:"
    )
    return prompt


def _build_jsquad_prompt(context: str, question: str, few_shot_examples: list[dict]) -> str:
    """Build a JSQuAD evaluation prompt."""
    prompt = ""
    for ex in few_shot_examples:
        prompt += (
            f"文脈: {ex['context']}\n"
            f"質問: {ex['question']}\n"
            f"回答: {ex['answer']}\n\n"
        )
    prompt += (
        f"文脈: {context}\n"
        f"質問: {question}\n"
        f"回答:"
    )
    return prompt


def _build_jcommonsenseqa_prompt(
    question: str, choices: list[str], few_shot_examples: list[dict]
) -> str:
    """Build a JCommonsenseQA evaluation prompt."""
    labels = ["A", "B", "C", "D", "E"]
    prompt = ""
    for ex in few_shot_examples:
        choice_str = " ".join(f"({labels[i]}) {c}" for i, c in enumerate(ex["choices"]))
        prompt += (
            f"質問: {ex['question']}\n"
            f"選択肢: {choice_str}\n"
            f"回答: {labels[ex['answer']]}\n\n"
        )
    choice_str = " ".join(f"({labels[i]}) {c}" for i, c in enumerate(choices))
    prompt += (
        f"質問: {question}\n"
        f"選択肢: {choice_str}\n"
        f"回答:"
    )
    return prompt


def _build_marc_prompt(text: str, few_shot_examples: list[dict]) -> str:
    """Build a MARC-ja evaluation prompt."""
    prompt = ""
    for ex in few_shot_examples:
        prompt += (
            f"レビュー: {ex['text']}\n"
            f"感情: {ex['label']}\n\n"
        )
    prompt += (
        f"レビュー: {text}\n"
        f"感情:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def _evaluate_jnli(
    model,
    tokenizer,
    n_shots: int = 0,
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate on JNLI (natural language inference)."""
    logger.info("Loading JNLI dataset...")
    dataset = load_dataset("shunk031/JGLUE", name="JNLI", split="validation")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Prepare few-shot examples from training set
    few_shot_examples = []
    if n_shots > 0:
        train_ds = load_dataset("shunk031/JGLUE", name="JNLI", split="train")
        for i in range(min(n_shots, len(train_ds))):
            ex = train_ds[i]
            few_shot_examples.append({
                "premise": ex["sentence1"],
                "hypothesis": ex["sentence2"],
                "label": JNLI_LABEL_MAP[ex["label"]],
            })

    correct = 0
    total = 0
    label_names = list(JNLI_LABEL_MAP.values())

    for sample in tqdm(dataset, desc="JNLI"):
        premise = sample["sentence1"]
        hypothesis = sample["sentence2"]
        gold_label = JNLI_LABEL_MAP[sample["label"]]

        prompt = _build_jnli_prompt(premise, hypothesis, few_shot_examples)
        pred = _generate_text(model, tokenizer, prompt, max_new_tokens=16).lower()

        # Match the first valid label in the output
        pred_label = None
        for lbl in label_names:
            if lbl in pred:
                pred_label = lbl
                break

        if pred_label == gold_label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("JNLI accuracy: %.4f (%d/%d)", accuracy, correct, total)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def _evaluate_jsquad(
    model,
    tokenizer,
    n_shots: int = 0,
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate on JSQuAD (question answering)."""
    logger.info("Loading JSQuAD dataset...")
    dataset = load_dataset("shunk031/JGLUE", name="JSQuAD", split="validation")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Prepare few-shot examples
    few_shot_examples = []
    if n_shots > 0:
        train_ds = load_dataset("shunk031/JGLUE", name="JSQuAD", split="train")
        for i in range(min(n_shots, len(train_ds))):
            ex = train_ds[i]
            few_shot_examples.append({
                "context": ex["context"],
                "question": ex["question"],
                "answer": ex["answers"]["text"][0],
            })

    f1_scores = []
    em_scores = []

    for sample in tqdm(dataset, desc="JSQuAD"):
        context = sample["context"]
        question = sample["question"]
        gold_answers = sample["answers"]["text"]

        prompt = _build_jsquad_prompt(context, question, few_shot_examples)
        pred = _generate_text(model, tokenizer, prompt, max_new_tokens=128)

        # Compute best score across all reference answers
        best_f1 = max(_char_f1(pred, ans) for ans in gold_answers)
        best_em = max(_exact_match(pred, ans) for ans in gold_answers)
        f1_scores.append(best_f1)
        em_scores.append(best_em)

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    logger.info("JSQuAD F1: %.4f, EM: %.4f (%d samples)", mean_f1, mean_em, len(f1_scores))
    return {"f1": mean_f1, "exact_match": mean_em, "total": len(f1_scores)}


def _evaluate_jcommonsenseqa(
    model,
    tokenizer,
    n_shots: int = 0,
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate on JCommonsenseQA (multiple choice)."""
    logger.info("Loading JCommonsenseQA dataset...")
    dataset = load_dataset("shunk031/JGLUE", name="JCommonsenseQA", split="validation")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    labels = ["A", "B", "C", "D", "E"]

    # Prepare few-shot examples
    few_shot_examples = []
    if n_shots > 0:
        train_ds = load_dataset("shunk031/JGLUE", name="JCommonsenseQA", split="train")
        for i in range(min(n_shots, len(train_ds))):
            ex = train_ds[i]
            few_shot_examples.append({
                "question": ex["question"],
                "choices": [ex[f"choice{j}"] for j in range(5)],
                "answer": ex["label"],
            })

    correct = 0
    total = 0

    for sample in tqdm(dataset, desc="JCommonsenseQA"):
        question = sample["question"]
        choices = [sample[f"choice{j}"] for j in range(5)]
        gold = sample["label"]  # int 0-4

        prompt = _build_jcommonsenseqa_prompt(question, choices, few_shot_examples)
        pred = _generate_text(model, tokenizer, prompt, max_new_tokens=8).strip().upper()

        # Extract predicted label
        pred_idx = None
        for i, lbl in enumerate(labels):
            if pred.startswith(lbl):
                pred_idx = i
                break

        if pred_idx == gold:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("JCommonsenseQA accuracy: %.4f (%d/%d)", accuracy, correct, total)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def _evaluate_marc_ja(
    model,
    tokenizer,
    n_shots: int = 0,
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate on MARC-ja (sentiment classification)."""
    logger.info("Loading MARC-ja dataset...")
    dataset = load_dataset("shunk031/JGLUE", name="MARC-ja", split="validation")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Prepare few-shot examples
    few_shot_examples = []
    if n_shots > 0:
        train_ds = load_dataset("shunk031/JGLUE", name="MARC-ja", split="train")
        for i in range(min(n_shots, len(train_ds))):
            ex = train_ds[i]
            few_shot_examples.append({
                "text": ex["sentence"],
                "label": MARC_LABEL_MAP[ex["label"]],
            })

    correct = 0
    total = 0

    for sample in tqdm(dataset, desc="MARC-ja"):
        text = sample["sentence"]
        gold_label = MARC_LABEL_MAP[sample["label"]]

        prompt = _build_marc_prompt(text, few_shot_examples)
        pred = _generate_text(model, tokenizer, prompt, max_new_tokens=16).lower()

        pred_label = None
        if "positive" in pred or "ポジティブ" in pred:
            pred_label = "positive"
        elif "negative" in pred or "ネガティブ" in pred:
            pred_label = "negative"

        if pred_label == gold_label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("MARC-ja accuracy: %.4f (%d/%d)", accuracy, correct, total)
    return {"accuracy": accuracy, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_jglue(
    model,
    tokenizer,
    tasks: Optional[list[str]] = None,
    n_shots: int = 0,
    max_samples: Optional[int] = None,
) -> dict:
    """Run JGLUE evaluation across all (or selected) tasks.

    Args:
        model: HuggingFace-compatible causal language model.
        tokenizer: Corresponding tokenizer.
        tasks: List of task names to evaluate. Defaults to all JGLUE tasks.
        n_shots: Number of few-shot examples.
        max_samples: Maximum samples per task (for quick testing).

    Returns:
        Dictionary with per-task scores and an aggregate score.
    """
    if tasks is None:
        tasks = list(JGLUE_TASKS)

    task_runners = {
        "jnli": _evaluate_jnli,
        "jsquad": _evaluate_jsquad,
        "jcommonsenseqa": _evaluate_jcommonsenseqa,
        "marc_ja": _evaluate_marc_ja,
    }

    results: dict = {}
    scores: list[float] = []

    for task in tasks:
        if task not in task_runners:
            logger.warning("Unknown JGLUE task: %s", task)
            continue
        task_result = task_runners[task](model, tokenizer, n_shots=n_shots, max_samples=max_samples)
        results[task] = task_result

        # Collect the primary score for aggregation
        if "accuracy" in task_result:
            scores.append(task_result["accuracy"])
        elif "f1" in task_result:
            scores.append(task_result["f1"])

    # Aggregate: simple average of primary scores
    aggregate = sum(scores) / len(scores) if scores else 0.0
    results["aggregate_score"] = aggregate

    logger.info("JGLUE aggregate score: %.4f", aggregate)
    return results
