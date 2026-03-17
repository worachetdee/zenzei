"""Unified evaluation entry point for the Zensei project.

Usage:
    python -m zensei.eval.run_eval --model_path checkpoints/zensei-v1 \
        --benchmarks jglue,perplexity,tokenizer_eval
    python -m zensei.eval.run_eval --model_path checkpoints/zensei-v1 --benchmarks all
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ALL_BENCHMARKS = ["jglue", "jaquad", "perplexity", "tokenizer_eval", "mmlu_ja"]


def _load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> tuple:
    """Load model and tokenizer from a HuggingFace-compatible checkpoint.

    Returns:
        (model, tokenizer) tuple.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info("Loading model from %s (dtype=%s, device=%s)", model_path, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def run(
    model_path: str,
    benchmarks: str = "all",
    output_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "bfloat16",
    n_shots: int = 0,
    max_samples: Optional[int] = None,
    base_tokenizer_path: Optional[str] = None,
) -> dict:
    """Run evaluation benchmarks for a Zensei model.

    Args:
        model_path: Path to the HuggingFace-compatible model checkpoint.
        benchmarks: Comma-separated list of benchmarks to run, or "all".
        output_path: Path to save JSON results. Defaults to
            ``{model_path}/eval_results.json``.
        device: Device map for model loading (default "auto").
        dtype: Model dtype — one of float32, float16, bfloat16.
        n_shots: Number of few-shot examples (where applicable).
        max_samples: Maximum number of samples per benchmark (for quick testing).
        base_tokenizer_path: Path to the base (DeepSeek) tokenizer for comparison
            in tokenizer_eval. If not provided, tokenizer_eval is skipped unless
            explicitly requested.

    Returns:
        Dictionary of benchmark results.
    """
    # ------------------------------------------------------------------ #
    # Resolve benchmark list
    # ------------------------------------------------------------------ #
    if benchmarks == "all":
        benchmark_list = list(ALL_BENCHMARKS)
    else:
        benchmark_list = [b.strip() for b in benchmarks.split(",")]
        for b in benchmark_list:
            if b not in ALL_BENCHMARKS:
                raise ValueError(
                    f"Unknown benchmark '{b}'. Choose from: {ALL_BENCHMARKS}"
                )

    logger.info("Benchmarks to run: %s", benchmark_list)

    # ------------------------------------------------------------------ #
    # Load model and tokenizer
    # ------------------------------------------------------------------ #
    model, tokenizer = _load_model_and_tokenizer(model_path, device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Run benchmarks
    # ------------------------------------------------------------------ #
    results: dict = {"model_path": model_path, "benchmarks": {}}
    overall_start = time.time()

    for bench_name in benchmark_list:
        logger.info("=" * 60)
        logger.info("Running benchmark: %s", bench_name)
        logger.info("=" * 60)
        bench_start = time.time()

        try:
            if bench_name == "jglue":
                from zensei.eval.jglue import evaluate_jglue

                bench_results = evaluate_jglue(
                    model, tokenizer, n_shots=n_shots, max_samples=max_samples
                )

            elif bench_name == "jaquad":
                from zensei.eval.jaquad import evaluate_jaquad

                bench_results = evaluate_jaquad(
                    model, tokenizer, n_shots=n_shots, max_samples=max_samples
                )

            elif bench_name == "perplexity":
                from zensei.eval.perplexity import evaluate_perplexity

                bench_results = evaluate_perplexity(
                    model, tokenizer, max_samples=max_samples
                )

            elif bench_name == "tokenizer_eval":
                from zensei.eval.tokenizer_eval import evaluate_tokenizer

                bench_results = evaluate_tokenizer(
                    tokenizer,
                    base_tokenizer_path=base_tokenizer_path,
                )

            elif bench_name == "mmlu_ja":
                from zensei.eval.mmlu_ja import evaluate_mmlu_ja

                bench_results = evaluate_mmlu_ja(
                    model, tokenizer, n_shots=n_shots, max_samples=max_samples
                )

            else:
                logger.warning("No handler for benchmark: %s", bench_name)
                bench_results = {"error": "not implemented"}

        except Exception as e:
            logger.error("Benchmark %s failed: %s", bench_name, e, exc_info=True)
            bench_results = {"error": str(e)}

        elapsed = time.time() - bench_start
        bench_results["elapsed_seconds"] = round(elapsed, 2)
        results["benchmarks"][bench_name] = bench_results
        logger.info("Benchmark %s completed in %.1f seconds", bench_name, elapsed)

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    total_elapsed = time.time() - overall_start
    results["total_elapsed_seconds"] = round(total_elapsed, 2)

    summary: dict = {}
    for name, res in results["benchmarks"].items():
        if "error" not in res:
            # Pull aggregate score if available
            for key in ("aggregate_score", "accuracy", "mean_perplexity", "overall_accuracy"):
                if key in res:
                    summary[name] = res[key]
                    break
    results["summary"] = summary

    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    for name, score in summary.items():
        logger.info("  %-20s : %.4f", name, score)
    logger.info("Total time: %.1f seconds", total_elapsed)

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    if output_path is None:
        output_path = str(Path(model_path) / "eval_results.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)

    return results


if __name__ == "__main__":
    fire.Fire(run)
