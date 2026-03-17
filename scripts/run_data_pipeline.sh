#!/bin/bash
set -euo pipefail
echo "=== Downloading corpora ==="
python -m zensei.data.download --output_dir data/raw
echo "=== Cleaning ==="
python -m zensei.data.clean --input_dir data/raw --output_dir data/cleaned
echo "=== Deduplication ==="
python -m zensei.data.dedup --input_dir data/cleaned --output_dir data/deduped
echo "=== Filtering ==="
python -m zensei.data.filter --input_dir data/deduped --output_dir data/filtered
echo "=== Preparing binary ==="
python -m zensei.data.prepare --input_dir data/filtered --output_dir data/processed --tokenizer_path data/tokenizer/zensei_merged
