"""
Phase 4 - Data mixing and curriculum scheduling.

Combines multiple binary datasets according to configurable mixing weights,
with optional curriculum scheduling to change weights across training steps.

YAML config example (mix_config.yaml):
    sources:
      wikipedia:
        bin_path: data/bin/wikipedia.bin
        idx_path: data/bin/wikipedia.idx
        weight: 0.3
      oscar:
        bin_path: data/bin/oscar.bin
        idx_path: data/bin/oscar.idx
        weight: 0.4
      mc4:
        bin_path: data/bin/mc4.bin
        idx_path: data/bin/mc4.idx
        weight: 0.3
    curriculum:
      - until_step: 10000
        weights: {wikipedia: 0.5, oscar: 0.3, mc4: 0.2}
      - until_step: 50000
        weights: {wikipedia: 0.3, oscar: 0.4, mc4: 0.3}
      - until_step: null
        weights: {wikipedia: 0.2, oscar: 0.4, mc4: 0.4}
    max_seq_len: 4096

Usage:
    python -m zensei.data.mix --config mix_config.yaml --output_prefix data/mixed/train
    python -m zensei.data.mix --config mix_config.yaml --mode dataloader --batch_size 8
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import fire
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Binary dataset reader
# ---------------------------------------------------------------------------


class BinaryDataset:
    """Read a memory-mapped binary dataset with its index."""

    def __init__(self, bin_path: str, idx_path: str) -> None:
        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)

        with open(self.idx_path, "r") as f:
            self.index = json.load(f)

        self.num_documents = self.index["num_documents"]
        self.total_tokens = self.index["total_tokens"]
        self.boundaries = self.index["document_boundaries"]

        self.data = np.memmap(
            str(self.bin_path), dtype=np.uint16, mode="r",
            shape=(self.total_tokens,),
        )

    def get_document(self, doc_idx: int) -> np.ndarray:
        """Return token ids for a single document."""
        start = self.boundaries[doc_idx]
        end = self.boundaries[doc_idx + 1]
        return np.array(self.data[start:end], dtype=np.int64)

    def __len__(self) -> int:
        return self.num_documents

    def __repr__(self) -> str:
        return (
            f"BinaryDataset({self.bin_path.name}, "
            f"docs={self.num_documents:,}, tokens={self.total_tokens:,})"
        )


# ---------------------------------------------------------------------------
# Mixing weights & curriculum
# ---------------------------------------------------------------------------


def get_weights_for_step(
    curriculum: Optional[List[Dict[str, Any]]],
    default_weights: Dict[str, float],
    step: int,
) -> Dict[str, float]:
    """Resolve mixing weights for a given training step.

    If curriculum is defined, find the matching stage. Otherwise use
    the default weights from the source config.
    """
    if not curriculum:
        return default_weights

    for stage in curriculum:
        until_step = stage.get("until_step")
        if until_step is None or step < until_step:
            return stage["weights"]

    # Past all stages, use the last one
    return curriculum[-1]["weights"]


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number")
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Mixed sampler
# ---------------------------------------------------------------------------


class MixedDataSampler:
    """Sample documents from multiple datasets according to mixing weights."""

    def __init__(
        self,
        config_path: str,
        seed: int = 42,
    ) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.sources: Dict[str, BinaryDataset] = {}
        self.default_weights: Dict[str, float] = {}

        for name, src_cfg in self.config["sources"].items():
            self.sources[name] = BinaryDataset(
                bin_path=src_cfg["bin_path"],
                idx_path=src_cfg["idx_path"],
            )
            self.default_weights[name] = src_cfg.get("weight", 1.0)

        self.curriculum = self.config.get("curriculum")
        self.max_seq_len = self.config.get("max_seq_len", 4096)
        self.rng = np.random.RandomState(seed)

        # Per-source document index cursors (shuffled)
        self._doc_indices: Dict[str, np.ndarray] = {}
        self._cursors: Dict[str, int] = {}
        for name, ds in self.sources.items():
            indices = np.arange(len(ds))
            self.rng.shuffle(indices)
            self._doc_indices[name] = indices
            self._cursors[name] = 0

        print(f"MixedDataSampler initialized with {len(self.sources)} sources:")
        for name, ds in self.sources.items():
            print(f"  {name}: {ds}")
        print(f"  Default weights: {self.default_weights}")
        if self.curriculum:
            print(f"  Curriculum stages: {len(self.curriculum)}")

    def _next_doc_from(self, source_name: str) -> np.ndarray:
        """Get the next document from a specific source (cycling with shuffle)."""
        ds = self.sources[source_name]
        indices = self._doc_indices[source_name]
        cursor = self._cursors[source_name]

        if cursor >= len(indices):
            # Reshuffle and reset
            self.rng.shuffle(indices)
            cursor = 0

        doc_idx = indices[cursor]
        self._cursors[source_name] = cursor + 1
        return ds.get_document(doc_idx)

    def sample(self, step: int = 0) -> Tuple[str, np.ndarray]:
        """Sample a single document according to current mixing weights.

        Returns:
            (source_name, token_ids)
        """
        weights = get_weights_for_step(
            self.curriculum, self.default_weights, step
        )
        weights = normalize_weights(weights)

        names = list(weights.keys())
        probs = [weights[n] for n in names]

        chosen = self.rng.choice(len(names), p=probs)
        source_name = names[chosen]
        tokens = self._next_doc_from(source_name)

        # Truncate to max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]

        return source_name, tokens

    def iter_batches(
        self,
        batch_size: int = 8,
        max_steps: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield batches of packed sequences.

        Each batch is a dict with:
          - input_ids: np.ndarray of shape (batch_size, max_seq_len)
          - sources: list of source names
          - step: current step number
        """
        step = 0
        while True:
            if max_steps is not None and step >= max_steps:
                break

            batch_ids = []
            batch_sources = []

            for _ in range(batch_size):
                source_name, tokens = self.sample(step=step)
                # Pad to max_seq_len
                padded = np.zeros(self.max_seq_len, dtype=np.int64)
                length = min(len(tokens), self.max_seq_len)
                padded[:length] = tokens[:length]
                batch_ids.append(padded)
                batch_sources.append(source_name)

            yield {
                "input_ids": np.stack(batch_ids),
                "sources": batch_sources,
                "step": step,
            }
            step += 1


# ---------------------------------------------------------------------------
# Combined output
# ---------------------------------------------------------------------------


def create_mixed_dataset(
    config_path: str,
    output_prefix: str,
    num_samples: int = 100_000,
    seed: int = 42,
) -> None:
    """Sample from multiple datasets and write a combined binary file.

    Args:
        config_path: Path to YAML mixing config.
        output_prefix: Output prefix for .bin and .idx files.
        num_samples: Number of documents to sample.
        seed: Random seed.
    """
    sampler = MixedDataSampler(config_path, seed=seed)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")

    print(f"\nCreating mixed dataset: {num_samples:,} samples -> {output_prefix}")

    all_tokens: List[np.ndarray] = []
    source_counts: Dict[str, int] = {}
    t0 = time.time()

    for i in range(num_samples):
        source_name, tokens = sampler.sample(step=i)
        all_tokens.append(tokens)
        source_counts[source_name] = source_counts.get(source_name, 0) + 1

        if (i + 1) % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  Sampled {i + 1:,} / {num_samples:,} "
                  f"({(i + 1) / elapsed:,.0f} samples/s)")

    # Build boundaries and concatenate
    boundaries = [0]
    for tokens in all_tokens:
        boundaries.append(boundaries[-1] + len(tokens))

    total_tokens = boundaries[-1]
    mmap = np.memmap(str(bin_path), dtype=np.uint16, mode="w+", shape=(total_tokens,))
    offset = 0
    for tokens in all_tokens:
        length = len(tokens)
        mmap[offset : offset + length] = tokens.astype(np.uint16)
        offset += length
    mmap.flush()
    del mmap

    index = {
        "num_documents": num_samples,
        "total_tokens": total_tokens,
        "dtype": "uint16",
        "document_boundaries": boundaries,
        "source_distribution": source_counts,
    }
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - t0
    size_mb = bin_path.stat().st_size / (1024 * 1024)

    print(f"\nMixed dataset written:")
    print(f"  Documents:  {num_samples:,}")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Size:       {size_mb:,.1f} MB")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"  Source distribution:")
    for name, count in sorted(source_counts.items()):
        pct = count / num_samples * 100
        print(f"    {name}: {count:,} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Dataloader demo
# ---------------------------------------------------------------------------


def run_dataloader(
    config_path: str,
    batch_size: int = 8,
    max_steps: int = 10,
    seed: int = 42,
) -> None:
    """Run a demo dataloader with mixed sampling.

    Args:
        config_path: Path to YAML mixing config.
        batch_size: Batch size.
        max_steps: Number of batches to produce.
        seed: Random seed.
    """
    sampler = MixedDataSampler(config_path, seed=seed)

    print(f"\nDataloader demo: batch_size={batch_size}, max_steps={max_steps}\n")

    for batch in sampler.iter_batches(batch_size=batch_size, max_steps=max_steps):
        step = batch["step"]
        ids = batch["input_ids"]
        sources = batch["sources"]
        non_zero = (ids != 0).sum()
        print(
            f"  Step {step:4d} | shape={ids.shape} | "
            f"non-zero tokens={non_zero:,} | "
            f"sources={sources}"
        )

    print("\nDataloader demo complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def mix(
    config: str = "mix_config.yaml",
    output_prefix: Optional[str] = None,
    mode: str = "dataset",
    num_samples: int = 100_000,
    batch_size: int = 8,
    max_steps: int = 10,
    seed: int = 42,
) -> None:
    """Data mixing and curriculum scheduling.

    Modes:
      - dataset: Create a combined binary dataset by sampling.
      - dataloader: Run a demo dataloader that streams mixed batches.

    Args:
        config: Path to YAML mixing configuration.
        output_prefix: Output prefix for binary files (dataset mode).
        mode: 'dataset' or 'dataloader'.
        num_samples: Number of documents to sample (dataset mode).
        batch_size: Batch size (dataloader mode).
        max_steps: Number of batches (dataloader mode).
        seed: Random seed.
    """
    if mode == "dataset":
        if not output_prefix:
            print("--output_prefix is required for dataset mode.")
            return
        create_mixed_dataset(config, output_prefix, num_samples, seed)
    elif mode == "dataloader":
        run_dataloader(config, batch_size, max_steps, seed)
    else:
        print(f"Unknown mode '{mode}'. Use 'dataset' or 'dataloader'.")


if __name__ == "__main__":
    fire.Fire(mix)
