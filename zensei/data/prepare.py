"""
Phase 4 - Tokenize documents and pack into memory-mapped binary format.

Produces two files per dataset:
  - {name}.bin  : uint16 numpy memmap of packed token ids
  - {name}.idx  : JSON index mapping document boundaries

Similar to Megatron-LM binary format but simplified.

Usage:
    python -m zensei.data.prepare --input_path data/filtered/wikipedia.jsonl \
                                  --output_prefix data/bin/wikipedia \
                                  --tokenizer_path tokenizer/merged
    python -m zensei.data.prepare --input_dir data/filtered \
                                  --output_dir data/bin \
                                  --tokenizer_path tokenizer/merged
"""

import json
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import fire
import numpy as np

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

# Global tokenizer handle (set in worker init)
_tokenizer = None


def _init_worker(tokenizer_path: str) -> None:
    """Initialize tokenizer in each worker process."""
    global _tokenizer
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _tokenize_texts(
    texts: List[str],
    max_seq_len: int,
) -> List[List[int]]:
    """Tokenize a batch of texts, truncating to max_seq_len."""
    global _tokenizer
    results = []
    for text in texts:
        ids = _tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        if ids:
            results.append(ids)
    return results


# ---------------------------------------------------------------------------
# Binary packing
# ---------------------------------------------------------------------------


def pack_to_binary(
    all_token_ids: List[List[int]],
    output_prefix: str,
) -> dict:
    """Pack tokenized documents into a memory-mapped binary file.

    Creates:
      {output_prefix}.bin  - flat uint16 array of all tokens
      {output_prefix}.idx  - JSON with document boundaries and metadata

    Args:
        all_token_ids: List of token id lists (one per document).
        output_prefix: Output file prefix (without extension).

    Returns:
        Statistics dict.
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")

    # Compute document boundaries (cumulative token counts)
    boundaries = [0]
    for ids in all_token_ids:
        boundaries.append(boundaries[-1] + len(ids))

    total_tokens = boundaries[-1]
    num_docs = len(all_token_ids)

    # Write binary file as memory-mapped uint16
    mmap = np.memmap(str(bin_path), dtype=np.uint16, mode="w+", shape=(total_tokens,))
    offset = 0
    for ids in all_token_ids:
        length = len(ids)
        mmap[offset : offset + length] = np.array(ids, dtype=np.uint16)
        offset += length
    mmap.flush()
    del mmap

    # Write index file
    index = {
        "num_documents": num_docs,
        "total_tokens": total_tokens,
        "dtype": "uint16",
        "document_boundaries": boundaries,
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f)

    size_mb = bin_path.stat().st_size / (1024 * 1024)

    stats = {
        "bin_path": str(bin_path),
        "idx_path": str(idx_path),
        "num_documents": num_docs,
        "total_tokens": total_tokens,
        "size_mb": round(size_mb, 2),
    }
    return stats


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------


def prepare_file(
    input_path: str,
    output_prefix: str,
    tokenizer_path: str,
    max_seq_len: int = 4096,
    num_workers: int = 0,
    batch_size: int = 1000,
) -> dict:
    """Tokenize a JSONL file and pack into binary format.

    Args:
        input_path: Input JSONL file.
        output_prefix: Output prefix (e.g. data/bin/wikipedia).
        tokenizer_path: Path to merged tokenizer.
        max_seq_len: Maximum sequence length in tokens (default 4096).
        num_workers: Number of parallel workers (0 = auto).
        batch_size: Documents per tokenization batch.

    Returns:
        Statistics dict.
    """
    input_path = Path(input_path)

    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    print(f"Preparing {input_path} -> {output_prefix}.[bin|idx]")
    print(f"  tokenizer={tokenizer_path}, max_seq_len={max_seq_len}, "
          f"workers={num_workers}")

    # Read all texts
    texts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(texts):,} documents")
    t0 = time.time()

    # Tokenize in parallel
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    worker_fn = partial(_tokenize_texts, max_seq_len=max_seq_len)

    all_token_ids: List[List[int]] = []

    if num_workers == 1:
        _init_worker(tokenizer_path)
        for batch in batches:
            results = worker_fn(batch)
            all_token_ids.extend(results)
    else:
        with Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(tokenizer_path,),
        ) as pool:
            for results in pool.imap(worker_fn, batches):
                all_token_ids.extend(results)
                if len(all_token_ids) % 50_000 < batch_size:
                    elapsed = time.time() - t0
                    print(f"  Tokenized {len(all_token_ids):,} docs "
                          f"({len(all_token_ids) / elapsed:,.0f} docs/s)")

    tokenize_time = time.time() - t0
    print(f"  Tokenization complete: {len(all_token_ids):,} docs in "
          f"{tokenize_time:.1f}s")

    # Pack into binary
    t1 = time.time()
    stats = pack_to_binary(all_token_ids, output_prefix)
    pack_time = time.time() - t1

    stats["input"] = str(input_path)
    stats["tokenize_time_s"] = round(tokenize_time, 2)
    stats["pack_time_s"] = round(pack_time, 2)
    stats["total_time_s"] = round(tokenize_time + pack_time, 2)

    print(f"  Binary written: {stats['num_documents']:,} docs, "
          f"{stats['total_tokens']:,} tokens, {stats['size_mb']:.1f} MB")
    print(f"  Total time: {stats['total_time_s']:.1f}s")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def prepare(
    input_path: Optional[str] = None,
    output_prefix: Optional[str] = None,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    tokenizer_path: str = "tokenizer/merged",
    max_seq_len: int = 4096,
    num_workers: int = 0,
    batch_size: int = 1000,
) -> None:
    """Tokenize JSONL and pack into memory-mapped binary format.

    Provide either (input_path, output_prefix) for a single file,
    or (input_dir, output_dir) to process all .jsonl files.

    Args:
        input_path: Single input JSONL file.
        output_prefix: Output file prefix (without extension).
        input_dir: Directory containing input JSONL files.
        output_dir: Directory for binary output files.
        tokenizer_path: Path to HuggingFace tokenizer directory.
        max_seq_len: Maximum sequence length (default 4096).
        num_workers: Parallel workers (0 = auto).
        batch_size: Documents per tokenization batch.
    """
    if input_path and output_prefix:
        prepare_file(
            input_path, output_prefix, tokenizer_path,
            max_seq_len, num_workers, batch_size,
        )
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
            prefix = out_dir / fp.stem
            stats = prepare_file(
                str(fp), str(prefix), tokenizer_path,
                max_seq_len, num_workers, batch_size,
            )
            all_stats.append(stats)
            print()

        # Summary
        total_docs = sum(s["num_documents"] for s in all_stats)
        total_tokens = sum(s["total_tokens"] for s in all_stats)
        total_mb = sum(s["size_mb"] for s in all_stats)
        print("=== Preparation Summary ===")
        print(f"  Files processed:  {len(all_stats)}")
        print(f"  Total documents:  {total_docs:,}")
        print(f"  Total tokens:     {total_tokens:,}")
        print(f"  Total size:       {total_mb:,.1f} MB")
    else:
        print("Provide either (--input_path, --output_prefix) or "
              "(--input_dir, --output_dir).")


if __name__ == "__main__":
    fire.Fire(prepare)
