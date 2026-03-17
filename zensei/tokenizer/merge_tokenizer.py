"""Merge a Japanese SentencePiece vocabulary into the DeepSeek base tokenizer.

Usage:
    python -m zensei.tokenizer.merge_tokenizer \
        --base_tokenizer deepseek-ai/DeepSeek-V3 \
        --ja_sp_model data/tokenizer/zensei_ja_sp.model \
        --output_dir data/tokenizer/zensei_merged

    python -m zensei.tokenizer.merge_tokenizer --config configs/tokenizer/tokenizer_train.yaml
"""

import logging
import os
from pathlib import Path
from typing import Optional

import fire
import sentencepiece as spm
import yaml
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge(
    config: Optional[str] = None,
    base_tokenizer: Optional[str] = None,
    ja_sp_model: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_new_tokens: int = 16128,
) -> None:
    """Merge Japanese SentencePiece tokens into the DeepSeek base tokenizer.

    The merged vocabulary is padded so that the final size is aligned to 128
    (129,280 base + up to 16,128 new = 145,408).

    Args:
        config: Path to YAML configuration file.
        base_tokenizer: HuggingFace model name or path for the base tokenizer.
        ja_sp_model: Path to the trained Japanese SentencePiece .model file.
        output_dir: Directory to save the merged tokenizer.
        max_new_tokens: Maximum number of new tokens to add (default 16128).
    """
    # ------------------------------------------------------------------ #
    # Resolve configuration
    # ------------------------------------------------------------------ #
    cfg: dict = {}
    if config is not None:
        logger.info("Loading config from %s", config)
        cfg = load_config(config)

    base_tokenizer = base_tokenizer or cfg.get("base_tokenizer", "deepseek-ai/DeepSeek-V3")
    ja_sp_model = ja_sp_model or cfg.get("model_prefix", "data/tokenizer/zensei_ja_sp")
    if ja_sp_model and not ja_sp_model.endswith(".model"):
        ja_sp_model = ja_sp_model + ".model"
    output_dir = output_dir or cfg.get("output_dir", "data/tokenizer/zensei_merged")
    max_new_tokens = cfg.get("max_new_tokens", max_new_tokens) if config else max_new_tokens

    # ------------------------------------------------------------------ #
    # Load base tokenizer
    # ------------------------------------------------------------------ #
    logger.info("Loading base tokenizer: %s", base_tokenizer)
    base_tok = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)
    base_vocab: dict[str, int] = base_tok.get_vocab()
    base_vocab_size = len(base_vocab)
    logger.info("Base tokenizer vocab size: %d", base_vocab_size)

    # ------------------------------------------------------------------ #
    # Load Japanese SentencePiece model
    # ------------------------------------------------------------------ #
    logger.info("Loading Japanese SP model: %s", ja_sp_model)
    sp = spm.SentencePieceProcessor()
    sp.load(ja_sp_model)
    ja_vocab_size = sp.get_piece_size()
    logger.info("Japanese SP vocab size: %d", ja_vocab_size)

    # ------------------------------------------------------------------ #
    # Extract new tokens not present in the base vocabulary
    # ------------------------------------------------------------------ #
    base_vocab_lower = set(base_vocab.keys())
    new_tokens: list[str] = []
    overlap_count = 0

    for i in range(ja_vocab_size):
        token = sp.id_to_piece(i)
        # Skip special tokens that already exist
        if token in ("<pad>", "<eos>", "<bos>", "<unk>", "<s>", "</s>"):
            continue
        if token in base_vocab_lower:
            overlap_count += 1
        else:
            new_tokens.append(token)

    logger.info("Tokens overlapping with base vocab: %d", overlap_count)
    logger.info("Candidate new tokens: %d", len(new_tokens))

    # ------------------------------------------------------------------ #
    # Cap the number of new tokens
    # ------------------------------------------------------------------ #
    if len(new_tokens) > max_new_tokens:
        logger.info(
            "Capping new tokens from %d to %d (max_new_tokens)",
            len(new_tokens),
            max_new_tokens,
        )
        new_tokens = new_tokens[:max_new_tokens]

    # ------------------------------------------------------------------ #
    # Align final vocab size to 128
    # ------------------------------------------------------------------ #
    target_vocab_size = base_vocab_size + len(new_tokens)
    remainder = target_vocab_size % 128
    if remainder != 0:
        padding_needed = 128 - remainder
        # Add dummy padding tokens to reach alignment
        for i in range(padding_needed):
            pad_token = f"<|pad_{i}|>"
            if pad_token not in base_vocab_lower:
                new_tokens.append(pad_token)
        logger.info(
            "Added %d padding tokens for 128-alignment (target %d -> %d)",
            padding_needed,
            target_vocab_size,
            base_vocab_size + len(new_tokens),
        )

    # ------------------------------------------------------------------ #
    # Add new tokens to the base tokenizer
    # ------------------------------------------------------------------ #
    num_added = base_tok.add_tokens(new_tokens)
    final_vocab_size = len(base_tok)

    logger.info("Tokens actually added: %d", num_added)
    logger.info("Final vocab size: %d", final_vocab_size)

    # ------------------------------------------------------------------ #
    # Save merged tokenizer
    # ------------------------------------------------------------------ #
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_tok.save_pretrained(output_dir)
    logger.info("Merged tokenizer saved to %s", output_dir)

    # Verify saved files
    expected_files = ["tokenizer.json", "tokenizer_config.json"]
    saved_files = os.listdir(output_dir)
    for ef in expected_files:
        status = "OK" if ef in saved_files else "MISSING"
        logger.info("  %s: %s", ef, status)

    # ------------------------------------------------------------------ #
    # Report statistics
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Merge Statistics")
    logger.info("=" * 60)
    logger.info("  Base vocab size     : %d", base_vocab_size)
    logger.info("  Japanese SP vocab   : %d", ja_vocab_size)
    logger.info("  Overlap count       : %d", overlap_count)
    logger.info("  New tokens added    : %d", num_added)
    logger.info("  Final vocab size    : %d", final_vocab_size)
    logger.info("  Aligned to 128      : %s", "Yes" if final_vocab_size % 128 == 0 else "No")
    logger.info("  Output directory    : %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(merge)
