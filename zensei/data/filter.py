"""
Phase 4 - Quality filtering for Japanese text.

Applies the following quality filters:
  1. Minimum / maximum document length (character count)
  2. Language detection confidence (CJK character ratio heuristic)
  3. Repetition filtering (repeated n-gram detection)
  4. Mean line length filtering

Usage:
    python -m zensei.data.filter --input_path data/dedup/wikipedia.jsonl \
                                 --output_path data/filtered/wikipedia.jsonl
    python -m zensei.data.filter --input_dir data/dedup --output_dir data/filtered
"""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import fire

# ---------------------------------------------------------------------------
# CJK detection (shared heuristic)
# ---------------------------------------------------------------------------


def _is_cjk(char: str) -> bool:
    """Check if a character is CJK / Hiragana / Katakana."""
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
        or (0x3040 <= cp <= 0x309F)
        or (0x30A0 <= cp <= 0x30FF)
        or (0x31F0 <= cp <= 0x31FF)
        or (0xFF65 <= cp <= 0xFF9F)
    )


def cjk_ratio(text: str) -> float:
    """Fraction of characters that are CJK."""
    if not text:
        return 0.0
    return sum(1 for ch in text if _is_cjk(ch)) / len(text)


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------


def check_length(text: str, min_len: int, max_len: int) -> bool:
    """Check document length is within bounds."""
    n = len(text)
    return min_len <= n <= max_len


def check_cjk_confidence(text: str, min_ratio: float) -> bool:
    """Check CJK character ratio as a proxy for Japanese language confidence."""
    return cjk_ratio(text) >= min_ratio


def check_repetition(
    text: str,
    ngram_size: int = 10,
    max_repeat_ratio: float = 0.2,
) -> bool:
    """Detect excessive repetition via n-gram analysis.

    If any single n-gram accounts for more than `max_repeat_ratio` of all
    n-grams, the document is considered repetitive.
    """
    if len(text) < ngram_size:
        return True  # too short to judge

    ngrams = [text[i : i + ngram_size] for i in range(len(text) - ngram_size + 1)]
    total = len(ngrams)
    if total == 0:
        return True

    counts = Counter(ngrams)
    most_common_count = counts.most_common(1)[0][1]
    return (most_common_count / total) <= max_repeat_ratio


def check_mean_line_length(
    text: str,
    min_mean: float = 10.0,
    max_mean: float = 10_000.0,
) -> bool:
    """Check that the mean line length is within acceptable range."""
    lines = text.split("\n")
    lines = [l for l in lines if l.strip()]  # skip blank lines
    if not lines:
        return False
    mean_len = sum(len(l) for l in lines) / len(lines)
    return min_mean <= mean_len <= max_mean


# ---------------------------------------------------------------------------
# Per-document filtering
# ---------------------------------------------------------------------------


def filter_document(
    text: str,
    min_len: int = 50,
    max_len: int = 500_000,
    min_cjk_ratio: float = 0.3,
    ngram_size: int = 10,
    max_repeat_ratio: float = 0.2,
    min_mean_line_len: float = 10.0,
    max_mean_line_len: float = 10_000.0,
) -> Optional[str]:
    """Return the rejection reason, or None if the document passes."""

    if not check_length(text, min_len, max_len):
        return "length"
    if not check_cjk_confidence(text, min_cjk_ratio):
        return "cjk_ratio"
    if not check_repetition(text, ngram_size, max_repeat_ratio):
        return "repetition"
    if not check_mean_line_length(text, min_mean_line_len, max_mean_line_len):
        return "mean_line_length"
    return None


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------


def filter_file(
    input_path: str,
    output_path: str,
    min_len: int = 50,
    max_len: int = 500_000,
    min_cjk_ratio: float = 0.3,
    ngram_size: int = 10,
    max_repeat_ratio: float = 0.2,
    min_mean_line_len: float = 10.0,
    max_mean_line_len: float = 10_000.0,
) -> dict:
    """Apply quality filters to a JSONL file.

    Returns:
        Statistics dict.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Filtering {input_path} -> {output_path}")

    total = 0
    kept = 0
    rejection_reasons: Counter = Counter()
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

            reason = filter_document(
                text,
                min_len=min_len,
                max_len=max_len,
                min_cjk_ratio=min_cjk_ratio,
                ngram_size=ngram_size,
                max_repeat_ratio=max_repeat_ratio,
                min_mean_line_len=min_mean_line_len,
                max_mean_line_len=max_mean_line_len,
            )

            if reason is not None:
                rejection_reasons[reason] += 1
            else:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

            if total % 50_000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {total:,} | Kept {kept:,} | {total / elapsed:,.0f} docs/s")

    elapsed = time.time() - t0
    filtered = total - kept

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "total": total,
        "kept": kept,
        "filtered": filtered,
        "filter_rate": filtered / total if total else 0,
        "rejection_reasons": dict(rejection_reasons),
        "elapsed_s": round(elapsed, 2),
    }

    print(f"\n  Total: {total:,} | Kept: {kept:,} | Filtered: {filtered:,} "
          f"({stats['filter_rate']:.1%})")
    print(f"  Rejection breakdown:")
    for reason, count in rejection_reasons.most_common():
        print(f"    {reason}: {count:,}")
    print(f"  Time: {elapsed:.1f}s")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def filter(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    min_len: int = 50,
    max_len: int = 500_000,
    min_cjk_ratio: float = 0.3,
    ngram_size: int = 10,
    max_repeat_ratio: float = 0.2,
    min_mean_line_len: float = 10.0,
    max_mean_line_len: float = 10_000.0,
) -> None:
    """Quality filtering for JSONL files.

    Provide either (input_path, output_path) for a single file,
    or (input_dir, output_dir) to process all .jsonl files in a directory.
    """
    kwargs = dict(
        min_len=min_len,
        max_len=max_len,
        min_cjk_ratio=min_cjk_ratio,
        ngram_size=ngram_size,
        max_repeat_ratio=max_repeat_ratio,
        min_mean_line_len=min_mean_line_len,
        max_mean_line_len=max_mean_line_len,
    )

    if input_path and output_path:
        filter_file(input_path, output_path, **kwargs)
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
            stats = filter_file(str(fp), str(op), **kwargs)
            all_stats.append(stats)
            print()

        # Summary
        total_in = sum(s["total"] for s in all_stats)
        total_kept = sum(s["kept"] for s in all_stats)
        total_filt = sum(s["filtered"] for s in all_stats)
        print("=== Filtering Summary ===")
        print(f"  Files processed: {len(all_stats)}")
        print(f"  Total docs:      {total_in:,}")
        print(f"  Kept:            {total_kept:,}")
        print(f"  Filtered:        {total_filt:,}")
        if total_in:
            print(f"  Overall filter rate: {total_filt / total_in:.1%}")

        # Aggregate rejection reasons
        agg_reasons: Counter = Counter()
        for s in all_stats:
            for reason, count in s["rejection_reasons"].items():
                agg_reasons[reason] += count
        print(f"  Rejection breakdown (all files):")
        for reason, count in agg_reasons.most_common():
            print(f"    {reason}: {count:,}")
    else:
        print("Provide either (--input_path, --output_path) or "
              "(--input_dir, --output_dir).")


if __name__ == "__main__":
    fire.Fire(filter)
