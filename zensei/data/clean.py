"""
Phase 4 - Japanese text cleaning pipeline.

Applies the following transformations:
  1. NFKC normalization
  2. HTML tag removal
  3. URL removal
  4. Control character removal
  5. CJK ratio filtering
  6. Punctuation normalization (half-width -> full-width)
  7. Whitespace normalization

Usage:
    python -m zensei.data.clean --input_path data/raw/wikipedia.jsonl \
                                --output_path data/clean/wikipedia.jsonl
    python -m zensei.data.clean --input_dir data/raw --output_dir data/clean
"""

import json
import os
import re
import time
import unicodedata
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

import fire

# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Half-width -> full-width Japanese punctuation mapping
_PUNCT_MAP = str.maketrans(
    {
        "!": "\uff01",  # !
        "?": "\uff1f",  # ?
        ",": "\u3001",  # ,
        ".": "\u3002",  # .
        ":": "\uff1a",  # :
        ";": "\uff1b",  # ;
        "(": "\uff08",  # (
        ")": "\uff09",  # )
    }
)


def _is_cjk(char: str) -> bool:
    """Check if a character is in a CJK Unicode block."""
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
        # Hiragana and Katakana
        or (0x3040 <= cp <= 0x309F)
        or (0x30A0 <= cp <= 0x30FF)
        or (0x31F0 <= cp <= 0x31FF)
        or (0xFF65 <= cp <= 0xFF9F)
    )


def cjk_ratio(text: str) -> float:
    """Return the fraction of characters that are CJK."""
    if not text:
        return 0.0
    total = len(text)
    cjk_count = sum(1 for ch in text if _is_cjk(ch))
    return cjk_count / total


def clean_text(text: str) -> str:
    """Apply all text-level cleaning steps."""
    # 1. NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove HTML tags
    text = _RE_HTML.sub("", text)

    # 3. Remove URLs
    text = _RE_URL.sub("", text)

    # 4. Remove control characters
    text = _RE_CONTROL.sub("", text)

    # 5. Punctuation normalization (half-width -> full-width)
    text = text.translate(_PUNCT_MAP)

    # 6. Whitespace normalization
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()

    return text


def process_line(line: str, min_cjk_ratio: float = 0.3) -> Optional[str]:
    """Clean a single JSONL line. Returns None if filtered out."""
    line = line.strip()
    if not line:
        return None

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None

    text = record.get("text", "")
    if not text:
        return None

    text = clean_text(text)

    if not text:
        return None

    # CJK ratio filter
    if cjk_ratio(text) < min_cjk_ratio:
        return None

    record["text"] = text
    return json.dumps(record, ensure_ascii=False)


def _process_chunk(
    chunk: List[str], min_cjk_ratio: float
) -> List[str]:
    """Process a chunk of lines (used by multiprocessing workers)."""
    results = []
    for line in chunk:
        result = process_line(line, min_cjk_ratio=min_cjk_ratio)
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------


def clean_file(
    input_path: str,
    output_path: str,
    min_cjk_ratio: float = 0.3,
    num_workers: int = 0,
    chunk_size: int = 10_000,
) -> dict:
    """Clean a single JSONL file.

    Args:
        input_path: Path to input JSONL.
        output_path: Path to write cleaned JSONL.
        min_cjk_ratio: Minimum CJK character ratio to keep a document.
        num_workers: Number of parallel workers (0 = auto).
        chunk_size: Lines per chunk for parallel processing.

    Returns:
        Statistics dict.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    print(f"Cleaning {input_path} -> {output_path}")
    print(f"  min_cjk_ratio={min_cjk_ratio}, workers={num_workers}")

    total = 0
    kept = 0
    t0 = time.time()

    # Read all lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)

    # Split into chunks
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

    worker_fn = partial(_process_chunk, min_cjk_ratio=min_cjk_ratio)

    with open(output_path, "w", encoding="utf-8") as out_f:
        if num_workers == 1:
            for chunk in chunks:
                results = worker_fn(chunk)
                for r in results:
                    out_f.write(r + "\n")
                    kept += 1
        else:
            with Pool(num_workers) as pool:
                for results in pool.imap(worker_fn, chunks):
                    for r in results:
                        out_f.write(r + "\n")
                        kept += 1

    elapsed = time.time() - t0
    filtered = total - kept
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "total": total,
        "kept": kept,
        "filtered": filtered,
        "filter_rate": filtered / total if total else 0,
        "elapsed_s": round(elapsed, 2),
    }

    print(f"  Total: {total:,} | Kept: {kept:,} | Filtered: {filtered:,} "
          f"({stats['filter_rate']:.1%}) | Time: {elapsed:.1f}s")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def clean(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    min_cjk_ratio: float = 0.3,
    num_workers: int = 0,
    chunk_size: int = 10_000,
) -> None:
    """Clean Japanese text in JSONL files.

    Provide either (input_path, output_path) for a single file,
    or (input_dir, output_dir) to process all .jsonl files in a directory.

    Args:
        input_path: Single input JSONL file.
        output_path: Single output JSONL file.
        input_dir: Directory containing input JSONL files.
        output_dir: Directory for cleaned output JSONL files.
        min_cjk_ratio: Minimum CJK character ratio (default 0.3).
        num_workers: Number of parallel workers (0 = cpu_count - 1).
        chunk_size: Lines per processing chunk.
    """
    if input_path and output_path:
        clean_file(input_path, output_path, min_cjk_ratio, num_workers, chunk_size)
    elif input_dir and output_dir:
        in_dir = Path(input_dir)
        out_dir = Path(output_dir)
        files = sorted(in_dir.glob("*.jsonl"))
        if not files:
            print(f"No .jsonl files found in {in_dir}")
            return
        print(f"Found {len(files)} JSONL file(s) in {in_dir}\n")
        all_stats = []
        for fp in files:
            op = out_dir / fp.name
            stats = clean_file(str(fp), str(op), min_cjk_ratio, num_workers, chunk_size)
            all_stats.append(stats)
            print()

        # Summary
        total_in = sum(s["total"] for s in all_stats)
        total_kept = sum(s["kept"] for s in all_stats)
        total_filt = sum(s["filtered"] for s in all_stats)
        print("=== Cleaning Summary ===")
        print(f"  Files processed: {len(all_stats)}")
        print(f"  Total docs:      {total_in:,}")
        print(f"  Kept:            {total_kept:,}")
        print(f"  Filtered:        {total_filt:,} "
              f"({total_filt / total_in:.1%})" if total_in else "")
    else:
        print("Provide either (--input_path, --output_path) or "
              "(--input_dir, --output_dir).")


if __name__ == "__main__":
    fire.Fire(clean)
