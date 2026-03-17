"""
Phase 4 - Download Japanese corpora from HuggingFace.

Supported datasets:
  - wikipedia (language=ja)
  - oscar (language=ja)
  - mc4 (language=ja)
  - cc100-ja

Usage:
    python -m zensei.data.download --dataset wikipedia --output_dir data/raw
    python -m zensei.data.download --dataset all --output_dir data/raw
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import fire
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "wikipedia": {
        "path": "wikipedia",
        "name": "20220301.ja",
        "split": "train",
        "text_field": "text",
    },
    "oscar": {
        "path": "oscar-corpus/OSCAR-2301",
        "name": "ja",
        "split": "train",
        "text_field": "text",
        "trust_remote_code": True,
    },
    "mc4": {
        "path": "mc4",
        "name": "ja",
        "split": "train",
        "text_field": "text",
    },
    "cc100-ja": {
        "path": "cc100",
        "name": "ja",
        "split": "train",
        "text_field": "text",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_file_lines(path: Path) -> int:
    """Return the number of lines already written (for resume support)."""
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _download_single(
    dataset_name: str,
    output_dir: str,
    streaming: bool = True,
    max_docs: Optional[int] = None,
) -> None:
    """Download a single dataset and write to JSONL."""

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset_name]
    out_path = Path(output_dir) / f"{dataset_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: count already-written lines
    existing = _count_file_lines(out_path)
    if existing > 0:
        print(f"[{dataset_name}] Resuming from document {existing}")

    load_kwargs = {
        "path": cfg["path"],
        "split": cfg["split"],
        "streaming": streaming,
    }
    if cfg.get("name"):
        load_kwargs["name"] = cfg["name"]
    if cfg.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True

    print(f"[{dataset_name}] Loading dataset (streaming={streaming}) ...")
    ds = load_dataset(**load_kwargs)

    text_field = cfg["text_field"]
    written = 0
    skipped = 0
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as f:
        for idx, example in enumerate(ds):
            # Skip already-written docs for resume
            if idx < existing:
                skipped += 1
                continue

            text = example.get(text_field, "")
            if not text or not text.strip():
                continue

            record = {"text": text.strip(), "source": dataset_name}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % 10_000 == 0:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                print(
                    f"  [{dataset_name}] {written:,} docs written "
                    f"({rate:,.0f} docs/s)"
                )

            if max_docs is not None and written >= max_docs:
                print(f"  [{dataset_name}] Reached max_docs={max_docs}, stopping.")
                break

    elapsed = time.time() - t0
    print(
        f"[{dataset_name}] Done. "
        f"Wrote {written:,} docs in {elapsed:.1f}s "
        f"(skipped {skipped:,} already-existing). "
        f"Output: {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def download(
    dataset: str = "all",
    output_dir: str = "data/raw",
    streaming: bool = True,
    max_docs: Optional[int] = None,
) -> None:
    """Download Japanese corpora and save as JSONL.

    Args:
        dataset: Name of dataset to download, or 'all' for every corpus.
        output_dir: Directory to save JSONL files.
        streaming: Use streaming mode (recommended for large datasets).
        max_docs: Optional cap on number of documents per dataset.
    """
    targets = list(DATASET_CONFIGS.keys()) if dataset == "all" else [dataset]

    print(f"Downloading {len(targets)} dataset(s) -> {output_dir}")
    print(f"  Datasets: {targets}")
    print(f"  Streaming: {streaming}")
    if max_docs:
        print(f"  Max docs per dataset: {max_docs}")
    print()

    for name in targets:
        try:
            _download_single(name, output_dir, streaming=streaming, max_docs=max_docs)
        except Exception as e:
            print(f"[{name}] ERROR: {e}")
            print(f"[{name}] Skipping this dataset.\n")

    # Summary
    print("\n=== Download Summary ===")
    for name in targets:
        p = Path(output_dir) / f"{name}.jsonl"
        if p.exists():
            n = _count_file_lines(p)
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {name}: {n:,} docs, {size_mb:,.1f} MB")
        else:
            print(f"  {name}: not downloaded")


if __name__ == "__main__":
    fire.Fire(download)
