"""
Phase 4 - MinHash-based deduplication for Japanese text.

Uses character 5-gram shingling (no word boundaries needed for Japanese)
with datasketch MinHash / MinHashLSH for near-duplicate detection.
Processes data in a streaming fashion for memory efficiency.

Usage:
    python -m zensei.data.dedup --input_path data/clean/wikipedia.jsonl \
                                --output_path data/dedup/wikipedia.jsonl
    python -m zensei.data.dedup --input_dir data/clean --output_dir data/dedup
"""

import json
import time
from pathlib import Path
from typing import Optional

import fire
from datasketch import MinHash, MinHashLSH

# ---------------------------------------------------------------------------
# Shingling
# ---------------------------------------------------------------------------


def char_ngram_shingles(text: str, n: int = 5) -> set:
    """Generate character n-gram shingles from text."""
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_minhash(shingles: set, num_perm: int = 128) -> MinHash:
    """Build a MinHash signature from a set of shingles."""
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def dedup_file(
    input_path: str,
    output_path: str,
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_n: int = 5,
) -> dict:
    """Deduplicate a JSONL file using MinHash LSH.

    Processes line-by-line in a streaming fashion:
      1. Read each document, compute its MinHash.
      2. Query the LSH index for near-duplicates.
      3. If no duplicate found, insert into index and write to output.

    Args:
        input_path: Input JSONL file.
        output_path: Output deduplicated JSONL file.
        threshold: Jaccard similarity threshold for dedup (default 0.8).
        num_perm: Number of permutations for MinHash (default 128).
        shingle_n: Character n-gram size for shingling (default 5).

    Returns:
        Statistics dict.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Deduplicating {input_path} -> {output_path}")
    print(f"  threshold={threshold}, num_perm={num_perm}, shingle_n={shingle_n}")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    total = 0
    kept = 0
    duplicates = 0
    t0 = time.time()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            text = record.get("text", "")

            if not text:
                continue

            # Build MinHash for this document
            shingles = char_ngram_shingles(text, n=shingle_n)
            mh = build_minhash(shingles, num_perm=num_perm)

            # Query for near-duplicates
            candidates = lsh.query(mh)

            if candidates:
                duplicates += 1
            else:
                # Unique document - insert and write
                doc_key = f"doc_{total}"
                try:
                    lsh.insert(doc_key, mh)
                except ValueError:
                    # Duplicate key (shouldn't happen but be safe)
                    pass
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

            if total % 10_000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(
                    f"  Processed {total:,} | Kept {kept:,} | "
                    f"Dupes {duplicates:,} | {rate:,.0f} docs/s"
                )

    elapsed = time.time() - t0
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "total": total,
        "duplicates_removed": duplicates,
        "remaining": kept,
        "dedup_rate": duplicates / total if total else 0,
        "elapsed_s": round(elapsed, 2),
    }

    print(
        f"\n  Total: {total:,} | Duplicates removed: {duplicates:,} | "
        f"Remaining: {kept:,} ({stats['dedup_rate']:.1%} removed) | "
        f"Time: {elapsed:.1f}s"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def dedup(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_n: int = 5,
) -> None:
    """MinHash deduplication for JSONL files.

    Provide either (input_path, output_path) for a single file,
    or (input_dir, output_dir) to process all .jsonl files in a directory.

    Args:
        input_path: Single input JSONL file.
        output_path: Single output JSONL file.
        input_dir: Directory containing input JSONL files.
        output_dir: Directory for deduplicated output JSONL files.
        threshold: Jaccard similarity threshold (default 0.8).
        num_perm: Number of MinHash permutations (default 128).
        shingle_n: Character n-gram size (default 5).
    """
    if input_path and output_path:
        dedup_file(input_path, output_path, threshold, num_perm, shingle_n)
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
            stats = dedup_file(str(fp), str(op), threshold, num_perm, shingle_n)
            all_stats.append(stats)
            print()

        # Summary
        total_in = sum(s["total"] for s in all_stats)
        total_dupes = sum(s["duplicates_removed"] for s in all_stats)
        total_kept = sum(s["remaining"] for s in all_stats)
        print("=== Deduplication Summary ===")
        print(f"  Files processed:     {len(all_stats)}")
        print(f"  Total documents:     {total_in:,}")
        print(f"  Duplicates removed:  {total_dupes:,}")
        print(f"  Remaining:           {total_kept:,}")
        if total_in:
            print(f"  Overall dedup rate:  {total_dupes / total_in:.1%}")
    else:
        print("Provide either (--input_path, --output_path) or "
              "(--input_dir, --output_dir).")


if __name__ == "__main__":
    fire.Fire(dedup)
