"""JaQuAD (Japanese Question Answering Dataset) evaluation for the Zensei project.

Evaluates reading comprehension using character-level F1 and exact match metrics,
which are standard for Japanese QA (no word tokenization needed).

Reference: https://huggingface.co/datasets/SkelterLabsInc/JaQuAD

Usage:
    from zensei.eval.jaquad import evaluate_jaquad
    results = evaluate_jaquad(model, tokenizer, n_shots=3)
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


# ---------------------------------------------------------------------------
# Japanese text normalization helpers
# ---------------------------------------------------------------------------

def normalize_japanese(text: str) -> str:
    """Normalize Japanese text for QA comparison.

    Steps:
        1. NFKC unicode normalization (fullwidth → halfwidth, etc.)
        2. Strip leading/trailing whitespace
        3. Collapse all internal whitespace (spaces between Japanese characters
           are typically irrelevant for QA evaluation)
        4. Lowercase (for any embedded Latin characters)
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = text.lower()
    return text


def remove_punctuation_ja(text: str) -> str:
    """Remove common Japanese and ASCII punctuation for looser matching."""
    # Japanese punctuation: 。、！？「」『』（）・…ー〜
    ja_punct = r"[。、！？「」『』（）・…ー〜\.\,\!\?\:\;\'\"\-\(\)\[\]\{\}]"
    return re.sub(ja_punct, "", text)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def char_f1_score(prediction: str, reference: str) -> float:
    """Compute character-level F1 between prediction and reference.

    This is the standard metric for Japanese QA evaluation since
    word-level tokenization introduces segmentation ambiguity.

    Args:
        prediction: Model-generated answer string.
        reference: Gold-standard answer string.

    Returns:
        F1 score in [0, 1].
    """
    pred_chars = list(normalize_japanese(prediction))
    ref_chars = list(normalize_japanese(reference))

    if not ref_chars and not pred_chars:
        return 1.0
    if not ref_chars or not pred_chars:
        return 0.0

    # Count character overlap using multiset intersection
    pred_counter: dict[str, int] = {}
    for c in pred_chars:
        pred_counter[c] = pred_counter.get(c, 0) + 1

    ref_counter: dict[str, int] = {}
    for c in ref_chars:
        ref_counter[c] = ref_counter.get(c, 0) + 1

    common = 0
    for c, count in pred_counter.items():
        common += min(count, ref_counter.get(c, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_chars)
    recall = common / len(ref_chars)
    return 2.0 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, reference: str) -> float:
    """Compute exact match after normalization.

    Args:
        prediction: Model-generated answer string.
        reference: Gold-standard answer string.

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    return 1.0 if normalize_japanese(prediction) == normalize_japanese(reference) else 0.0


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """Generate an answer from a prompt using greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(
    context: str,
    question: str,
    few_shot_examples: list[dict],
) -> str:
    """Build a JaQuAD reading comprehension prompt.

    Args:
        context: Passage text.
        question: Question about the passage.
        few_shot_examples: List of dicts with keys: context, question, answer.

    Returns:
        Formatted prompt string.
    """
    prompt = "以下の文脈に基づいて質問に答えてください。回答は文脈中の表現をそのまま使ってください。\n\n"

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


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_jaquad(
    model,
    tokenizer,
    n_shots: int = 0,
    max_samples: Optional[int] = None,
    dataset_name: str = "SkelterLabsInc/JaQuAD",
    split: str = "validation",
) -> dict:
    """Evaluate a model on JaQuAD reading comprehension.

    Args:
        model: HuggingFace-compatible causal language model.
        tokenizer: Corresponding tokenizer.
        n_shots: Number of few-shot examples to prepend.
        max_samples: Maximum number of evaluation samples (None = all).
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to evaluate on.

    Returns:
        Dictionary with F1, exact match, and sample counts.
    """
    logger.info("Loading JaQuAD dataset (%s, split=%s)...", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Prepare few-shot examples from the training set
    few_shot_examples: list[dict] = []
    if n_shots > 0:
        train_ds = load_dataset(dataset_name, split="train")
        for i in range(min(n_shots, len(train_ds))):
            ex = train_ds[i]
            few_shot_examples.append({
                "context": ex["context"],
                "question": ex["question"],
                "answer": ex["answers"]["text"][0],
            })

    f1_scores: list[float] = []
    em_scores: list[float] = []

    for sample in tqdm(dataset, desc="JaQuAD"):
        context = sample["context"]
        question = sample["question"]
        gold_answers = sample["answers"]["text"]

        prompt = _build_prompt(context, question, few_shot_examples)
        prediction = _generate_answer(model, tokenizer, prompt)

        # Take the best score across all reference answers
        best_f1 = max(char_f1_score(prediction, ans) for ans in gold_answers)
        best_em = max(exact_match_score(prediction, ans) for ans in gold_answers)
        f1_scores.append(best_f1)
        em_scores.append(best_em)

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_em = sum(em_scores) / len(em_scores) if em_scores else 0.0

    logger.info("JaQuAD results — F1: %.4f, EM: %.4f (%d samples)", mean_f1, mean_em, len(f1_scores))

    return {
        "f1": round(mean_f1, 4),
        "exact_match": round(mean_em, 4),
        "total": len(f1_scores),
        "aggregate_score": round(mean_f1, 4),
    }
