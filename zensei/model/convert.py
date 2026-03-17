"""
Weight converter: DeepSeek-V3 checkpoint → Zensei format.

Handles:
  1. Loading sharded DeepSeek-V3 checkpoints (safetensors or PyTorch).
  2. Expanding embedding / lm_head weight matrices to accommodate the larger
     Zensei vocabulary (129 280 → 145 408) using mean + noise initialization
     (the Swallow approach).
  3. Saving the converted state dict as safetensors shards.

Usage (CLI via Fire):
    python -m zensei.model.convert \\
        --input_dir /data/deepseek-v3 \\
        --output_dir /data/zensei-671b \\
        --new_vocab_size 145408
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_deepseek_checkpoint(
    path: Union[str, Path],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a DeepSeek-V3 checkpoint, merging shards into one state dict.

    Supports both ``safetensors`` and PyTorch ``.bin`` / ``.pt`` shard formats.
    Shards are detected by globbing the directory for matching files.

    Args:
        path: Directory containing checkpoint shards.
        device: Device to map tensors to (default ``"cpu"``).

    Returns:
        Merged state dict mapping parameter names to tensors.
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {path}")

    state_dict: dict[str, torch.Tensor] = {}

    # Try safetensors first, then PyTorch bins
    safetensor_shards = sorted(path.glob("*.safetensors"))
    bin_shards = sorted(path.glob("*.bin")) + sorted(path.glob("*.pt"))

    if safetensor_shards:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load .safetensors checkpoints. "
                "Install with: pip install safetensors"
            ) from exc

        logger.info("Loading %d safetensors shard(s) from %s", len(safetensor_shards), path)
        for shard in tqdm(safetensor_shards, desc="Loading shards"):
            shard_dict = load_file(str(shard), device=device)
            state_dict.update(shard_dict)

    elif bin_shards:
        logger.info("Loading %d PyTorch shard(s) from %s", len(bin_shards), path)
        for shard in tqdm(bin_shards, desc="Loading shards"):
            shard_dict = torch.load(str(shard), map_location=device, weights_only=True)
            state_dict.update(shard_dict)
    else:
        raise FileNotFoundError(
            f"No checkpoint shards found in {path}. "
            "Expected *.safetensors, *.bin, or *.pt files."
        )

    logger.info("Loaded %d parameters from checkpoint.", len(state_dict))
    return state_dict


# ---------------------------------------------------------------------------
# Vocabulary expansion (Swallow approach)
# ---------------------------------------------------------------------------

def expand_vocab_weights(
    state_dict: dict[str, torch.Tensor],
    old_vocab: int = 129280,
    new_vocab: int = 145408,
    embed_key: str = "tok_emb.weight",
    lm_head_key: str = "lm_head.weight",
    noise_std: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Expand embedding and lm_head weight matrices for a larger vocabulary.

    New rows are initialized to the mean of the existing embeddings plus
    small Gaussian noise (the *Swallow* approach), which has been shown to
    give stable continued-pretraining for vocabulary-extended LLMs.

    Args:
        state_dict: Model state dict (modified in-place and returned).
        old_vocab: Original vocabulary size in the checkpoint.
        new_vocab: Target vocabulary size.
        embed_key: Key for the token embedding weight in the state dict.
        lm_head_key: Key for the language model head weight in the state dict.
        noise_std: Standard deviation of Gaussian noise added to new rows.

    Returns:
        The (modified) state dict.
    """
    if new_vocab <= old_vocab:
        logger.info("new_vocab (%d) <= old_vocab (%d); skipping expansion.", new_vocab, old_vocab)
        return state_dict

    extra = new_vocab - old_vocab
    logger.info("Expanding vocabulary from %d → %d (+%d tokens).", old_vocab, new_vocab, extra)

    for key in (embed_key, lm_head_key):
        if key not in state_dict:
            logger.warning("Key '%s' not found in state dict; skipping.", key)
            continue

        weight = state_dict[key]  # (old_vocab, dim)
        if weight.shape[0] != old_vocab:
            logger.warning(
                "Expected %s to have %d rows but found %d; skipping.",
                key, old_vocab, weight.shape[0],
            )
            continue

        dim = weight.shape[1]
        mean_embedding = weight.mean(dim=0, keepdim=True)  # (1, dim)
        noise = torch.randn(extra, dim, dtype=weight.dtype, device=weight.device) * noise_std
        new_rows = mean_embedding.expand(extra, -1) + noise

        state_dict[key] = torch.cat([weight, new_rows], dim=0)
        logger.info("  %s: %s → %s", key, tuple(weight.shape), tuple(state_dict[key].shape))

    return state_dict


# ---------------------------------------------------------------------------
# Full conversion pipeline
# ---------------------------------------------------------------------------

def convert_and_save(
    input_dir: str,
    output_dir: str,
    new_vocab_size: int = 145408,
    old_vocab_size: int = 129280,
    max_shard_size_gb: float = 5.0,
    save_format: str = "safetensors",
) -> None:
    """Load a DeepSeek-V3 checkpoint, expand the vocabulary, and save.

    Args:
        input_dir: Path to the directory with DeepSeek-V3 checkpoint shards.
        output_dir: Destination directory for converted Zensei checkpoint.
        new_vocab_size: Target vocabulary size (default 145 408).
        old_vocab_size: Source vocabulary size (default 129 280).
        max_shard_size_gb: Maximum shard size in GB when saving.
        save_format: Either ``"safetensors"`` or ``"pytorch"``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load
    logger.info("Step 1/3: Loading DeepSeek-V3 checkpoint from %s", input_dir)
    state_dict = load_deepseek_checkpoint(input_dir)

    # 2. Expand vocabulary
    logger.info("Step 2/3: Expanding vocabulary weights")
    state_dict = expand_vocab_weights(
        state_dict,
        old_vocab=old_vocab_size,
        new_vocab=new_vocab_size,
    )

    # 3. Save
    logger.info("Step 3/3: Saving converted checkpoint to %s", output_dir)
    _save_state_dict(state_dict, output_path, max_shard_size_gb, save_format)
    logger.info("Conversion complete.")


def _save_state_dict(
    state_dict: dict[str, torch.Tensor],
    output_dir: Path,
    max_shard_size_gb: float,
    save_format: str,
) -> None:
    """Save a state dict, optionally sharded."""
    max_shard_bytes = int(max_shard_size_gb * (1024 ** 3))

    if save_format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to save in safetensors format. "
                "Install with: pip install safetensors"
            ) from exc

        # Simple sharding by cumulative tensor size
        shards: list[dict[str, torch.Tensor]] = []
        current_shard: dict[str, torch.Tensor] = {}
        current_size = 0

        for name, tensor in state_dict.items():
            tensor_bytes = tensor.numel() * tensor.element_size()
            if current_size + tensor_bytes > max_shard_bytes and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            current_shard[name] = tensor
            current_size += tensor_bytes

        if current_shard:
            shards.append(current_shard)

        for idx, shard in enumerate(tqdm(shards, desc="Saving shards")):
            shard_name = f"model-{idx + 1:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, str(output_dir / shard_name))
            logger.info("  Saved %s (%d tensors)", shard_name, len(shard))

    else:
        # PyTorch format — single file
        out_path = output_dir / "model.bin"
        torch.save(state_dict, str(out_path))
        logger.info("Saved %s (%d tensors)", out_path.name, len(state_dict))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Fire CLI entry point."""
    import fire

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    fire.Fire({
        "convert": convert_and_save,
        "load": load_deepseek_checkpoint,
        "expand": expand_vocab_weights,
    })


if __name__ == "__main__":
    main()
