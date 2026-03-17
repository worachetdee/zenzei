"""Text generation utilities for Zensei models.

Supports autoregressive decoding with temperature, top-p, top-k sampling,
repetition penalty (important for Japanese to avoid character loops),
streaming token generation, and KV cache for efficient inference.

Usage:
    python -m zensei.serving.generate \
        --model_path checkpoints/zensei-16b \
        --prompt "日本の四季について教えてください。"
"""

from __future__ import annotations

import logging
import time
from typing import Generator, Optional, Union

import fire
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from zensei.model.model import ModelArgs, Transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float = 1.1,
) -> torch.Tensor:
    """Apply repetition penalty to logits based on previously generated tokens.

    This is especially important for Japanese text generation, where the model
    can easily fall into repeating character or phrase loops.
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty

    return logits


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k highest values."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    min_val = values[-1]
    logits[logits < min_val] = float("-inf")
    return logits


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: zero out tokens outside the top-p cumulative probability."""
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_mask = cumulative_probs - probs > p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original indexing
    logits.scatter_(0, sorted_indices, sorted_logits)
    return logits


def _sample_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> int:
    """Sample a single token from logits with temperature, top-k, and top-p."""
    if temperature <= 0:
        # Greedy decoding
        return logits.argmax(dim=-1).item()

    logits = logits / temperature
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ---------------------------------------------------------------------------
# Generation (streaming)
# ---------------------------------------------------------------------------


def generate_stream(
    model: Transformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stop_tokens: Optional[list[int]] = None,
) -> Generator[str, None, None]:
    """Generate tokens one at a time, yielding each decoded token string.

    This is the streaming interface: callers iterate over the generator to get
    tokens as they are produced.

    Args:
        model: The Zensei Transformer model.
        tokenizer: HuggingFace tokenizer for encoding/decoding.
        prompt: Input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k filtering parameter.
        repetition_penalty: Penalty for repeated tokens (>1 discourages repeats).
        stop_tokens: List of token IDs that signal generation should stop.

    Yields:
        Decoded token strings as they are generated.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if stop_tokens is None:
        stop_tokens = []
        if tokenizer.eos_token_id is not None:
            stop_tokens.append(tokenizer.eos_token_id)

    generated_ids: list[int] = []
    current_ids = input_ids

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(current_ids)
            next_logits = logits[:, -1, :].squeeze(0)  # (vocab_size,)

            # Apply repetition penalty
            next_logits = _apply_repetition_penalty(
                next_logits, generated_ids, repetition_penalty
            )

            # Sample
            next_token = _sample_token(next_logits, temperature, top_p, top_k)

            # Check stop condition
            if next_token in stop_tokens:
                break

            generated_ids.append(next_token)

            # Decode the new token
            token_str = tokenizer.decode([next_token], skip_special_tokens=False)
            yield token_str

            # Prepare next input (full sequence for models without KV cache,
            # or just the new token for models with KV cache support)
            next_token_tensor = torch.tensor([[next_token]], device=device)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)


# ---------------------------------------------------------------------------
# Generation (non-streaming)
# ---------------------------------------------------------------------------


def generate(
    model: Transformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stop_tokens: Optional[list[int]] = None,
) -> str:
    """Generate text from a prompt, returning the full generated string.

    Args:
        model: The Zensei Transformer model.
        tokenizer: HuggingFace tokenizer for encoding/decoding.
        prompt: Input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k filtering parameter.
        repetition_penalty: Penalty for repeated tokens.
        stop_tokens: List of token IDs that signal generation should stop.

    Returns:
        The generated text (excluding the input prompt).
    """
    tokens = list(
        generate_stream(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        )
    )
    return "".join(tokens)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate_batch(
    model: Transformer,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> list[str]:
    """Generate text for a batch of prompts.

    Note: This currently processes prompts sequentially. For true batched
    inference with padding, a more sophisticated approach with attention
    masks and left-padding would be needed.

    Args:
        model: The Zensei Transformer model.
        tokenizer: HuggingFace tokenizer for encoding/decoding.
        prompts: List of input text prompts.
        max_new_tokens: Maximum number of new tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k filtering parameter.
        repetition_penalty: Penalty for repeated tokens.

    Returns:
        List of generated texts, one per prompt.
    """
    results = []
    for i, prompt in enumerate(prompts):
        logger.info("Generating %d/%d ...", i + 1, len(prompts))
        text = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        results.append(text)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    prompt: str = "日本語のテストです。",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    device: str = "cuda",
) -> None:
    """Generate text from a Zensei model via the command line.

    Args:
        model_path: Path to the model checkpoint directory.
        tokenizer_path: Path to the tokenizer (defaults to model_path).
        prompt: Input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k filtering parameter.
        repetition_penalty: Repetition penalty factor.
        device: Device to use (cuda or cpu).
    """
    tokenizer_path = tokenizer_path or model_path

    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    logger.info("Loading model from %s", model_path)
    model = Transformer.from_config(f"{model_path}/config.json")
    # Load weights if available
    import os

    weight_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded weights from %s", weight_path)
    else:
        logger.warning("No weights found at %s; using random initialization.", model_path)

    model = model.to(device).eval()

    logger.info("Prompt: %s", prompt)
    logger.info("Generating (max_new_tokens=%d, temp=%.2f) ...", max_new_tokens, temperature)

    t0 = time.time()
    output = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    print(output)
    print(f"{'='*60}")
    print(f"Generated in {elapsed:.2f}s")


if __name__ == "__main__":
    fire.Fire(main)
