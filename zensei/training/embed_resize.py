"""Embedding matrix expansion for vocabulary extension (Swallow approach).

When extending a pretrained model's vocabulary with new tokens (e.g., adding
Japanese subwords to a base DeepSeek tokenizer), the embedding and LM-head
matrices must be resized accordingly.  New rows are initialized to the mean of
the existing embeddings plus a small Gaussian perturbation so that the new
tokens start in a reasonable region of the embedding space.

Usage (standalone):
    python -m zensei.training.embed_resize \
        --checkpoint_path checkpoints/base \
        --old_vocab_size 102400 \
        --new_vocab_size 128256
"""

import copy
import logging
from typing import Optional

import fire
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core resize helpers
# ---------------------------------------------------------------------------


def _expand_embedding_weight(
    old_weight: torch.Tensor,
    old_vocab_size: int,
    new_vocab_size: int,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """Create a new embedding weight tensor with expanded vocabulary.

    Args:
        old_weight: Original weight tensor of shape ``(old_vocab_size, dim)``.
        old_vocab_size: Number of tokens in the original vocabulary.
        new_vocab_size: Number of tokens in the expanded vocabulary.
        noise_std: Standard deviation of the Gaussian noise added to new rows.

    Returns:
        New weight tensor of shape ``(new_vocab_size, dim)`` with the original
        weights preserved exactly and new rows initialized to
        ``mean(old_weight) + N(0, noise_std)``.
    """
    assert new_vocab_size > old_vocab_size, (
        f"new_vocab_size ({new_vocab_size}) must be greater than "
        f"old_vocab_size ({old_vocab_size})"
    )

    dim = old_weight.shape[1]
    new_weight = torch.empty(new_vocab_size, dim, dtype=old_weight.dtype, device=old_weight.device)

    # Preserve original weights exactly
    new_weight[:old_vocab_size] = old_weight[:old_vocab_size]

    # Initialize new rows: mean of existing embeddings + small noise
    mean_embedding = old_weight[:old_vocab_size].float().mean(dim=0)
    num_new = new_vocab_size - old_vocab_size
    noise = torch.randn(num_new, dim, dtype=torch.float32, device=old_weight.device) * noise_std
    new_rows = (mean_embedding.unsqueeze(0) + noise).to(old_weight.dtype)
    new_weight[old_vocab_size:] = new_rows

    return new_weight


def resize_embeddings(
    model: nn.Module,
    old_vocab_size: int,
    new_vocab_size: int,
    noise_std: float = 0.01,
    embed_attr: str = "embed.weight",
    head_attr: str = "head.weight",
    inplace: bool = True,
) -> nn.Module:
    """Expand embedding and LM-head matrices for vocabulary extension.

    This implements the *Swallow* approach: new token embeddings are
    initialized to the mean of the original embeddings plus a small Gaussian
    perturbation, preserving the pretrained weights for all existing tokens.

    Args:
        model: The language model (must have ``embed`` and ``head`` sub-modules).
        old_vocab_size: Size of the original vocabulary.
        new_vocab_size: Size of the expanded vocabulary.
        noise_std: Std of Gaussian noise for new token initialization.
        embed_attr: Dot-separated attribute path to the input embedding weight.
        head_attr: Dot-separated attribute path to the output projection weight.
        inplace: If *True*, modify the model in place. Otherwise return a copy.

    Returns:
        The model with resized embedding layers.
    """
    if new_vocab_size <= old_vocab_size:
        logger.warning(
            "new_vocab_size (%d) <= old_vocab_size (%d); nothing to do.",
            new_vocab_size,
            old_vocab_size,
        )
        return model

    if not inplace:
        model = copy.deepcopy(model)

    logger.info(
        "Resizing embeddings: %d -> %d (adding %d new tokens)",
        old_vocab_size,
        new_vocab_size,
        new_vocab_size - old_vocab_size,
    )

    # --- Expand input embeddings ---
    _resize_parameter(model, embed_attr, old_vocab_size, new_vocab_size, noise_std)

    # --- Expand output projection / LM head ---
    _resize_parameter(model, head_attr, old_vocab_size, new_vocab_size, noise_std)

    logger.info("Embedding resize complete.")
    return model


def _resolve_attr(module: nn.Module, attr_path: str):
    """Resolve a dot-separated attribute path on a module."""
    parts = attr_path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _resize_parameter(
    model: nn.Module,
    attr_path: str,
    old_vocab_size: int,
    new_vocab_size: int,
    noise_std: float,
) -> None:
    """Resize a single parameter (embedding or head weight) in-place."""
    parent, name = _resolve_attr(model, attr_path)
    old_param = getattr(parent, name)

    if not isinstance(old_param, (torch.Tensor, nn.Parameter)):
        raise TypeError(f"Expected Tensor at '{attr_path}', got {type(old_param)}")

    old_weight = old_param.data if isinstance(old_param, nn.Parameter) else old_param
    new_weight = _expand_embedding_weight(old_weight, old_vocab_size, new_vocab_size, noise_std)

    if isinstance(old_param, nn.Parameter):
        new_param = nn.Parameter(new_weight, requires_grad=old_param.requires_grad)
        setattr(parent, name, new_param)
    else:
        setattr(parent, name, new_weight)

    logger.info(
        "  %s: (%d, %d) -> (%d, %d)",
        attr_path,
        old_vocab_size,
        old_weight.shape[1],
        new_vocab_size,
        new_weight.shape[1],
    )


# ---------------------------------------------------------------------------
# Optional: resize an nn.Embedding layer directly
# ---------------------------------------------------------------------------


def resize_embedding_layer(
    embedding: nn.Embedding,
    new_vocab_size: int,
    noise_std: float = 0.01,
) -> nn.Embedding:
    """Create a new ``nn.Embedding`` with expanded vocabulary.

    Preserves original weights and initializes new rows with the mean +
    Gaussian noise strategy.

    Args:
        embedding: The original embedding layer.
        new_vocab_size: Target vocabulary size.
        noise_std: Std of Gaussian noise for new rows.

    Returns:
        A new ``nn.Embedding`` with the expanded weight matrix.
    """
    old_vocab_size, dim = embedding.weight.shape
    if new_vocab_size <= old_vocab_size:
        return embedding

    new_weight = _expand_embedding_weight(
        embedding.weight.data, old_vocab_size, new_vocab_size, noise_std
    )
    new_embedding = nn.Embedding(
        new_vocab_size,
        dim,
        padding_idx=embedding.padding_idx,
    )
    new_embedding.weight = nn.Parameter(new_weight)
    return new_embedding


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    checkpoint_path: str,
    old_vocab_size: int,
    new_vocab_size: int,
    output_path: Optional[str] = None,
    noise_std: float = 0.01,
) -> None:
    """Resize embedding matrices in a saved checkpoint.

    Args:
        checkpoint_path: Path to a PyTorch checkpoint file (.pt / .bin).
        old_vocab_size: Original vocabulary size.
        new_vocab_size: Target vocabulary size.
        output_path: Where to save the resized checkpoint.
            Defaults to ``<checkpoint_path>.resized``.
        noise_std: Std of Gaussian noise for new rows.
    """
    if output_path is None:
        output_path = checkpoint_path + ".resized"

    logger.info("Loading checkpoint from %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Detect common key patterns for embed and head weights
    embed_key = None
    head_key = None
    for key in state_dict.keys():
        if "embed" in key.lower() and "weight" in key.lower() and embed_key is None:
            embed_key = key
        if ("head" in key.lower() or "lm_head" in key.lower()) and "weight" in key.lower():
            head_key = key

    if embed_key is None:
        raise KeyError("Could not find embedding weight in checkpoint keys.")
    if head_key is None:
        raise KeyError("Could not find LM head weight in checkpoint keys.")

    logger.info("Detected embed key: %s", embed_key)
    logger.info("Detected head key : %s", head_key)

    for key in [embed_key, head_key]:
        old_weight = state_dict[key]
        new_weight = _expand_embedding_weight(old_weight, old_vocab_size, new_vocab_size, noise_std)
        state_dict[key] = new_weight
        logger.info("  %s: %s -> %s", key, list(old_weight.shape), list(new_weight.shape))

    logger.info("Saving resized checkpoint to %s", output_path)
    torch.save(state_dict, output_path)
    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(main)
