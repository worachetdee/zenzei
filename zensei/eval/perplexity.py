"""Perplexity evaluation for the Zensei project.

Computes sliding-window perplexity on held-out Japanese text corpora.
Supports per-token and per-character perplexity with configurable
window size and stride.

Usage:
    from zensei.eval.perplexity import evaluate_perplexity
    results = evaluate_perplexity(model, tokenizer)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default test sets to evaluate on
DEFAULT_TEST_SETS = {
    "wikipedia": {
        "dataset": "wikipedia",
        "config": "20220301.ja",
        "split": "train",
        "text_field": "text",
        "description": "Japanese Wikipedia",
    },
    "news": {
        "dataset": "cc_news",
        "split": "train",
        "text_field": "text",
        "description": "CC-News (Japanese filtered)",
    },
    "literature": {
        "dataset": "range3/aozora-clean",
        "split": "train",
        "text_field": "text",
        "description": "Aozora Bunko (Japanese literature)",
    },
}


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    text: str,
    window_size: int = 1024,
    stride: int = 512,
    device: Optional[str] = None,
) -> dict:
    """Compute sliding-window perplexity for a single text.

    Uses a strided sliding window to handle texts longer than the model's
    context length. Only tokens that fall within the current window but
    outside the overlap with the previous window are scored, to avoid
    double-counting at the boundaries.

    Args:
        model: HuggingFace-compatible causal language model.
        tokenizer: Corresponding tokenizer.
        text: Input text to evaluate.
        window_size: Size of the sliding window in tokens.
        stride: Number of tokens to advance per step.
        device: Device to run on (inferred from model if None).

    Returns:
        Dictionary with token_perplexity, char_perplexity, n_tokens, n_chars.
    """
    if device is None:
        device = str(next(model.parameters()).device)

    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"].to(device)
    seq_len = input_ids.size(1)

    if seq_len == 0:
        return {
            "token_perplexity": float("inf"),
            "char_perplexity": float("inf"),
            "n_tokens": 0,
            "n_chars": 0,
            "total_nll": 0.0,
        }

    total_nll = 0.0
    total_tokens = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + window_size, seq_len)
        # The target range: only score tokens not in the overlap from the
        # previous window (except for the very first window)
        target_begin = max(begin, stride) if begin > 0 else 1
        target_len = end - target_begin

        if target_len <= 0:
            break

        window_ids = input_ids[:, begin:end]
        outputs = model(window_ids)
        # Handle both tuple and object outputs
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        # Shift for causal LM: predict token t from tokens 0..t-1
        # logits[:, t-1, :] predicts token at position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = window_ids[:, 1:].contiguous()

        # Compute per-token cross entropy
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Only accumulate for the target range
        # In the shifted indexing: position t in the labels corresponds to
        # predicting token at position t+1 in the window
        target_offset = target_begin - begin - 1  # offset in the shifted sequence
        target_losses = token_losses[target_offset : target_offset + target_len]

        total_nll += target_losses.sum().item()
        total_tokens += target_len

        if end >= seq_len:
            break

    # Per-token perplexity
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    token_ppl = math.exp(avg_nll) if avg_nll < 100 else float("inf")

    # Per-character perplexity: distribute total NLL over character count
    n_chars = len(text)
    char_nll = total_nll / n_chars if n_chars > 0 else float("inf")
    char_ppl = math.exp(char_nll) if char_nll < 100 else float("inf")

    return {
        "token_perplexity": token_ppl,
        "char_perplexity": char_ppl,
        "n_tokens": total_tokens,
        "n_chars": n_chars,
        "total_nll": total_nll,
    }


def _load_test_texts(
    test_set_name: str,
    test_set_config: dict,
    max_samples: int = 100,
    min_length: int = 200,
) -> list[str]:
    """Load text samples from a HuggingFace dataset.

    Args:
        test_set_name: Name identifier for logging.
        test_set_config: Configuration dict with dataset, split, text_field keys.
        max_samples: Maximum number of texts to load.
        min_length: Minimum character length for a text to be included.

    Returns:
        List of text strings.
    """
    logger.info("Loading test set '%s' (%s)...", test_set_name, test_set_config["description"])

    load_kwargs = {"split": test_set_config["split"], "streaming": True}
    if "config" in test_set_config:
        load_kwargs["name"] = test_set_config["config"]

    try:
        dataset = load_dataset(test_set_config["dataset"], **load_kwargs)
    except Exception as e:
        logger.warning("Could not load dataset '%s': %s", test_set_config["dataset"], e)
        return []

    texts: list[str] = []
    text_field = test_set_config["text_field"]

    for sample in dataset:
        text = sample.get(text_field, "")
        if len(text) >= min_length:
            texts.append(text)
        if len(texts) >= max_samples:
            break

    logger.info("Loaded %d texts from '%s'", len(texts), test_set_name)
    return texts


def evaluate_perplexity(
    model,
    tokenizer,
    test_sets: Optional[dict] = None,
    window_size: int = 1024,
    stride: int = 512,
    max_samples: Optional[int] = 100,
) -> dict:
    """Evaluate perplexity across multiple Japanese text corpora.

    Args:
        model: HuggingFace-compatible causal language model.
        tokenizer: Corresponding tokenizer.
        test_sets: Dictionary of test set configurations. Defaults to
            Wikipedia-ja, CC-News, and Aozora Bunko.
        window_size: Sliding window size in tokens.
        stride: Sliding window stride in tokens.
        max_samples: Maximum number of texts per test set.

    Returns:
        Dictionary with per-test-set and aggregate perplexity results.
    """
    if test_sets is None:
        test_sets = DEFAULT_TEST_SETS

    results: dict = {}
    all_token_ppls: list[float] = []
    all_char_ppls: list[float] = []

    for name, config in test_sets.items():
        texts = _load_test_texts(name, config, max_samples=max_samples or 100)
        if not texts:
            results[name] = {"error": "no data loaded"}
            continue

        token_ppls: list[float] = []
        char_ppls: list[float] = []

        for text in tqdm(texts, desc=f"Perplexity ({name})"):
            ppl_result = compute_perplexity(
                model, tokenizer, text,
                window_size=window_size, stride=stride,
            )
            if ppl_result["token_perplexity"] != float("inf"):
                token_ppls.append(ppl_result["token_perplexity"])
                char_ppls.append(ppl_result["char_perplexity"])

        if token_ppls:
            mean_token_ppl = sum(token_ppls) / len(token_ppls)
            std_token_ppl = (
                sum((x - mean_token_ppl) ** 2 for x in token_ppls) / len(token_ppls)
            ) ** 0.5
            mean_char_ppl = sum(char_ppls) / len(char_ppls)
            std_char_ppl = (
                sum((x - mean_char_ppl) ** 2 for x in char_ppls) / len(char_ppls)
            ) ** 0.5

            results[name] = {
                "mean_token_perplexity": round(mean_token_ppl, 2),
                "std_token_perplexity": round(std_token_ppl, 2),
                "mean_char_perplexity": round(mean_char_ppl, 2),
                "std_char_perplexity": round(std_char_ppl, 2),
                "n_samples": len(token_ppls),
            }
            all_token_ppls.extend(token_ppls)
            all_char_ppls.extend(char_ppls)

            logger.info(
                "%s — token PPL: %.2f (±%.2f), char PPL: %.2f (±%.2f), n=%d",
                name, mean_token_ppl, std_token_ppl, mean_char_ppl, std_char_ppl, len(token_ppls),
            )
        else:
            results[name] = {"error": "all samples returned inf perplexity"}

    # Overall aggregate
    if all_token_ppls:
        overall_token_ppl = sum(all_token_ppls) / len(all_token_ppls)
        overall_char_ppl = sum(all_char_ppls) / len(all_char_ppls)
        results["mean_perplexity"] = round(overall_token_ppl, 2)
        results["mean_char_perplexity"] = round(overall_char_ppl, 2)
        results["total_samples"] = len(all_token_ppls)
        logger.info(
            "Overall — token PPL: %.2f, char PPL: %.2f (%d total samples)",
            overall_token_ppl, overall_char_ppl, len(all_token_ppls),
        )
    else:
        results["mean_perplexity"] = float("inf")

    return results
