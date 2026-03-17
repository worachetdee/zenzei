"""Train a Japanese SentencePiece BPE tokenizer for the Zensei project.

Usage:
    python -m zensei.tokenizer.train_tokenizer --config configs/tokenizer/tokenizer_train.yaml
    python -m zensei.tokenizer.train_tokenizer --input_dir data/raw/ja_corpus --vocab_size 32000
"""

import glob
import logging
import os
import time
from pathlib import Path
from typing import Optional

import fire
import sentencepiece as spm
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def collect_input_files(input_dir: str) -> list[str]:
    """Collect all .txt files from the input directory recursively."""
    patterns = ["**/*.txt", "**/*.tsv", "**/*.csv"]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pattern), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No text files found in {input_dir}. "
            "Expected .txt, .tsv, or .csv files."
        )
    return files


def train(
    config: Optional[str] = None,
    input_dir: Optional[str] = None,
    model_prefix: Optional[str] = None,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    normalization_rule_name: str = "nfkc",
    byte_fallback: bool = True,
    split_digits: bool = True,
    max_sentence_length: int = 16384,
    num_threads: int = 16,
    train_extremely_large_corpus: bool = True,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
) -> None:
    """Train a Japanese SentencePiece BPE tokenizer.

    Args:
        config: Path to a YAML configuration file. CLI args override YAML values.
        input_dir: Directory containing the Japanese text corpus.
        model_prefix: Output path prefix for the trained model files.
        vocab_size: Target vocabulary size (default 32000).
        model_type: SentencePiece model type (default "bpe").
        character_coverage: Character coverage ratio (default 0.9995).
        normalization_rule_name: Unicode normalization rule (default "nfkc").
        byte_fallback: Enable byte-fallback for unknown characters.
        split_digits: Split digits into individual tokens.
        max_sentence_length: Maximum sentence length in bytes.
        num_threads: Number of training threads.
        train_extremely_large_corpus: Enable large-corpus training mode.
        pad_id: Token ID for <pad>.
        unk_id: Token ID for <unk>.
        bos_id: Token ID for <bos>.
        eos_id: Token ID for <eos>.
    """
    # ------------------------------------------------------------------ #
    # Merge YAML config with CLI overrides
    # ------------------------------------------------------------------ #
    cfg: dict = {}
    if config is not None:
        logger.info("Loading config from %s", config)
        cfg = load_config(config)

    # CLI arguments take precedence over YAML values
    input_dir = input_dir or cfg.get("input_dir", "data/raw/ja_corpus")
    model_prefix = model_prefix or cfg.get("model_prefix", "data/tokenizer/zensei_ja_sp")
    vocab_size = cfg.get("vocab_size", vocab_size) if config and "vocab_size" not in _cli_keys() else vocab_size
    model_type = cfg.get("model_type", model_type) if config else model_type
    character_coverage = cfg.get("character_coverage", character_coverage) if config else character_coverage
    normalization_rule_name = cfg.get("normalization_rule_name", normalization_rule_name) if config else normalization_rule_name
    byte_fallback = cfg.get("byte_fallback", byte_fallback) if config else byte_fallback
    split_digits = cfg.get("split_digits", split_digits) if config else split_digits
    max_sentence_length = cfg.get("max_sentence_length", max_sentence_length) if config else max_sentence_length
    num_threads = cfg.get("num_threads", num_threads) if config else num_threads
    train_extremely_large_corpus = cfg.get("train_extremely_large_corpus", train_extremely_large_corpus) if config else train_extremely_large_corpus
    pad_id = cfg.get("pad_id", pad_id) if config else pad_id
    unk_id = cfg.get("unk_id", unk_id) if config else unk_id
    bos_id = cfg.get("bos_id", bos_id) if config else bos_id
    eos_id = cfg.get("eos_id", eos_id) if config else eos_id

    # ------------------------------------------------------------------ #
    # Collect input files
    # ------------------------------------------------------------------ #
    input_files = collect_input_files(input_dir)
    total_size_mb = sum(os.path.getsize(f) for f in input_files) / (1024 * 1024)
    logger.info(
        "Found %d input files (%.2f MB total) in %s",
        len(input_files),
        total_size_mb,
        input_dir,
    )

    # ------------------------------------------------------------------ #
    # Ensure output directory exists
    # ------------------------------------------------------------------ #
    output_dir = os.path.dirname(model_prefix)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Train SentencePiece model
    # ------------------------------------------------------------------ #
    input_arg = ",".join(input_files)

    logger.info("Starting SentencePiece training ...")
    logger.info("  model_type        = %s", model_type)
    logger.info("  vocab_size        = %d", vocab_size)
    logger.info("  character_coverage = %.4f", character_coverage)
    logger.info("  normalization     = %s", normalization_rule_name)
    logger.info("  byte_fallback     = %s", byte_fallback)
    logger.info("  model_prefix      = %s", model_prefix)

    start_time = time.time()

    spm.SentencePieceTrainer.train(
        input=input_arg,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        normalization_rule_name=normalization_rule_name,
        byte_fallback=byte_fallback,
        split_digits=split_digits,
        max_sentence_length=max_sentence_length,
        num_threads=num_threads,
        train_extremely_large_corpus=train_extremely_large_corpus,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        # Special token surface forms
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
    )

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------ #
    # Log training statistics
    # ------------------------------------------------------------------ #
    model_path = f"{model_prefix}.model"
    vocab_path = f"{model_prefix}.vocab"

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    actual_vocab_size = sp.get_piece_size()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Elapsed time      : %.1f seconds", elapsed)
    logger.info("  Model file        : %s (%.2f MB)", model_path, model_size_mb)
    logger.info("  Vocab file        : %s", vocab_path)
    logger.info("  Actual vocab size : %d", actual_vocab_size)
    logger.info("  Corpus size       : %.2f MB across %d files", total_size_mb, len(input_files))
    logger.info("=" * 60)


def _cli_keys() -> set[str]:
    """Return an empty set -- helper to keep config-merge logic simple."""
    return set()


if __name__ == "__main__":
    fire.Fire(train)
