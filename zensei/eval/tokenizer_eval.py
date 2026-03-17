"""Tokenizer quality evaluation for the Zensei project.

Compares the Zensei merged tokenizer against the base DeepSeek tokenizer
across key efficiency metrics:
  - Fertility (tokens per character)
  - Compression ratio (bytes per token)
  - UNK rate (percentage of unknown tokens)
  - Roundtrip accuracy (encode -> decode exact match)

Tests across multiple Japanese text domains: Wikipedia, news, literary text,
and technical text.

Usage:
    from zensei.eval.tokenizer_eval import evaluate_tokenizer
    results = evaluate_tokenizer(zensei_tokenizer, base_tokenizer_path="deepseek-ai/deepseek-v3")
"""

from __future__ import annotations

import logging
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Test domains with their HuggingFace dataset sources
TEST_DOMAINS = {
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
        "description": "CC-News",
    },
    "literary": {
        "dataset": "range3/aozora-clean",
        "split": "train",
        "text_field": "text",
        "description": "Aozora Bunko (literary)",
    },
    "technical": {
        "dataset": "wikipedia",
        "config": "20220301.ja",
        "split": "train",
        "text_field": "text",
        "description": "Japanese Wikipedia (technical proxy)",
    },
}

# Target thresholds for Zensei tokenizer quality
TARGETS = {
    "fertility": 1.5,  # tokens per character — lower is better
    "unk_rate": 0.001,  # < 0.1% UNK tokens
    "roundtrip_accuracy": 0.999,  # > 99.9% encode-decode exact match
}


def _load_domain_texts(
    domain_name: str,
    domain_config: dict,
    max_samples: int = 200,
    min_length: int = 100,
) -> list[str]:
    """Load text samples from a HuggingFace dataset for a given domain."""
    logger.info("Loading domain '%s' (%s)...", domain_name, domain_config["description"])

    load_kwargs = {"split": domain_config["split"], "streaming": True}
    if "config" in domain_config:
        load_kwargs["name"] = domain_config["config"]

    try:
        dataset = load_dataset(domain_config["dataset"], **load_kwargs)
    except Exception as e:
        logger.warning("Could not load dataset for domain '%s': %s", domain_name, e)
        return []

    texts: list[str] = []
    text_field = domain_config["text_field"]

    for sample in dataset:
        text = sample.get(text_field, "")
        if len(text) >= min_length:
            # Truncate very long texts for efficiency
            texts.append(text[:5000])
        if len(texts) >= max_samples:
            break

    logger.info("Loaded %d texts for domain '%s'", len(texts), domain_name)
    return texts


def _compute_fertility(tokenizer, texts: list[str]) -> float:
    """Compute average fertility (tokens per character).

    Lower values mean better compression of Japanese text.
    Target for Japanese-optimized tokenizer: < 1.5.
    """
    total_tokens = 0
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_chars += len(text)

    return total_tokens / total_chars if total_chars > 0 else float("inf")


def _compute_compression_ratio(tokenizer, texts: list[str]) -> float:
    """Compute average compression ratio (bytes per token).

    Higher values mean each token covers more raw bytes — more efficient.
    """
    total_bytes = 0
    total_tokens = 0

    for text in texts:
        text_bytes = len(text.encode("utf-8"))
        tokens = tokenizer.encode(text)
        total_bytes += text_bytes
        total_tokens += len(tokens)

    return total_bytes / total_tokens if total_tokens > 0 else 0.0


def _compute_unk_rate(tokenizer, texts: list[str]) -> float:
    """Compute percentage of UNK tokens.

    A good tokenizer with byte-fallback should produce near-zero UNK rate.
    """
    total_tokens = 0
    total_unks = 0
    unk_id = getattr(tokenizer, "unk_token_id", None)

    if unk_id is None:
        # No UNK token defined — likely uses byte-fallback, so UNK rate is 0
        return 0.0

    for text in texts:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_unks += sum(1 for t in tokens if t == unk_id)

    return total_unks / total_tokens if total_tokens > 0 else 0.0


def _compute_roundtrip_accuracy(tokenizer, texts: list[str]) -> float:
    """Compute encode -> decode roundtrip exact match accuracy.

    Measures whether tokenization is lossless: encode(text) -> decode should
    produce the original text.
    """
    total = 0
    exact_matches = 0

    for text in texts:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        total += 1
        if decoded == text:
            exact_matches += 1

    return exact_matches / total if total > 0 else 0.0


def _evaluate_single_tokenizer(
    tokenizer,
    tokenizer_name: str,
    texts_by_domain: dict[str, list[str]],
) -> dict:
    """Evaluate a single tokenizer across all domains.

    Returns:
        Dictionary with per-domain and aggregate metrics.
    """
    results: dict = {}
    all_texts: list[str] = []

    for domain_name, texts in texts_by_domain.items():
        if not texts:
            results[domain_name] = {"error": "no data"}
            continue

        fertility = _compute_fertility(tokenizer, texts)
        compression = _compute_compression_ratio(tokenizer, texts)
        unk_rate = _compute_unk_rate(tokenizer, texts)
        roundtrip = _compute_roundtrip_accuracy(tokenizer, texts)

        results[domain_name] = {
            "fertility": round(fertility, 4),
            "compression_ratio": round(compression, 4),
            "unk_rate": round(unk_rate, 6),
            "roundtrip_accuracy": round(roundtrip, 4),
            "n_texts": len(texts),
        }
        all_texts.extend(texts)

    # Aggregate across all domains
    if all_texts:
        results["aggregate"] = {
            "fertility": round(_compute_fertility(tokenizer, all_texts), 4),
            "compression_ratio": round(_compute_compression_ratio(tokenizer, all_texts), 4),
            "unk_rate": round(_compute_unk_rate(tokenizer, all_texts), 6),
            "roundtrip_accuracy": round(_compute_roundtrip_accuracy(tokenizer, all_texts), 4),
            "n_texts": len(all_texts),
        }

    logger.info(
        "%s aggregate — fertility: %.4f, compression: %.4f, UNK: %.6f, roundtrip: %.4f",
        tokenizer_name,
        results.get("aggregate", {}).get("fertility", 0),
        results.get("aggregate", {}).get("compression_ratio", 0),
        results.get("aggregate", {}).get("unk_rate", 0),
        results.get("aggregate", {}).get("roundtrip_accuracy", 0),
    )

    return results


def _format_comparison_table(zensei_results: dict, base_results: dict | None) -> str:
    """Format a human-readable comparison table.

    Returns:
        Formatted string table.
    """
    lines: list[str] = []
    sep = "-" * 85

    lines.append(sep)
    lines.append(f"{'Metric':<25} {'Domain':<15} {'Zensei':>12}", )

    if base_results is not None:
        lines[-1] += f" {'Base':>12} {'Delta':>12}"

    lines.append(sep)

    domains = [d for d in zensei_results if d != "aggregate"]
    domains.append("aggregate")

    for domain in domains:
        z_data = zensei_results.get(domain, {})
        b_data = base_results.get(domain, {}) if base_results else {}

        if "error" in z_data:
            continue

        for metric in ["fertility", "compression_ratio", "unk_rate", "roundtrip_accuracy"]:
            z_val = z_data.get(metric, 0)
            line = f"{metric:<25} {domain:<15} {z_val:>12.4f}"

            if base_results is not None and metric in b_data:
                b_val = b_data[metric]
                delta = z_val - b_val
                sign = "+" if delta > 0 else ""
                line += f" {b_val:>12.4f} {sign}{delta:>11.4f}"

            lines.append(line)
        lines.append("")

    lines.append(sep)

    # Target comparison
    lines.append("\nTarget thresholds:")
    z_agg = zensei_results.get("aggregate", {})
    for metric, target in TARGETS.items():
        actual = z_agg.get(metric, 0)
        passed = ""
        if metric == "fertility":
            passed = "PASS" if actual < target else "FAIL"
        elif metric == "unk_rate":
            passed = "PASS" if actual < target else "FAIL"
        elif metric == "roundtrip_accuracy":
            passed = "PASS" if actual > target else "FAIL"
        lines.append(f"  {metric:<25} target: {target:<10} actual: {actual:<10.4f}  [{passed}]")

    return "\n".join(lines)


def evaluate_tokenizer(
    tokenizer,
    base_tokenizer_path: Optional[str] = None,
    domains: Optional[dict] = None,
    max_samples: int = 200,
) -> dict:
    """Evaluate tokenizer quality and optionally compare against a base tokenizer.

    Args:
        tokenizer: The Zensei (merged) tokenizer to evaluate.
        base_tokenizer_path: Path or HuggingFace ID for the base (DeepSeek)
            tokenizer. If provided, a comparison table is generated.
        domains: Dictionary of domain configurations. Defaults to
            Wikipedia-ja, news, literary, and technical text.
        max_samples: Maximum number of text samples per domain.

    Returns:
        Dictionary with evaluation results and comparison data.
    """
    if domains is None:
        domains = TEST_DOMAINS

    # Load texts for all domains
    texts_by_domain: dict[str, list[str]] = {}
    for domain_name, config in domains.items():
        texts_by_domain[domain_name] = _load_domain_texts(
            domain_name, config, max_samples=max_samples
        )

    # Evaluate Zensei tokenizer
    logger.info("Evaluating Zensei tokenizer...")
    zensei_results = _evaluate_single_tokenizer(tokenizer, "Zensei", texts_by_domain)

    # Evaluate base tokenizer if provided
    base_results = None
    if base_tokenizer_path is not None:
        logger.info("Loading base tokenizer from %s...", base_tokenizer_path)
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(
                base_tokenizer_path, trust_remote_code=True
            )
            logger.info("Evaluating base tokenizer...")
            base_results = _evaluate_single_tokenizer(
                base_tokenizer, "Base (DeepSeek)", texts_by_domain
            )
        except Exception as e:
            logger.error("Failed to load base tokenizer: %s", e)

    # Format comparison table
    table = _format_comparison_table(zensei_results, base_results)
    logger.info("\n%s", table)

    # Build result dict
    result: dict = {
        "zensei": zensei_results,
        "comparison_table": table,
    }
    if base_results is not None:
        result["base"] = base_results

    # Provide an aggregate score (fertility is the primary metric for tokenizer quality)
    agg = zensei_results.get("aggregate", {})
    result["fertility"] = agg.get("fertility", 0)
    result["compression_ratio"] = agg.get("compression_ratio", 0)
    result["unk_rate"] = agg.get("unk_rate", 0)
    result["roundtrip_accuracy"] = agg.get("roundtrip_accuracy", 0)

    return result
