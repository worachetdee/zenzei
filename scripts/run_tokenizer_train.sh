#!/bin/bash
set -euo pipefail
python -m zensei.tokenizer.train_tokenizer --config configs/tokenizer/tokenizer_train.yaml
python -m zensei.tokenizer.merge_tokenizer --config configs/tokenizer/tokenizer_train.yaml
python -m zensei.tokenizer.test_tokenizer --tokenizer_path data/tokenizer/zensei_merged
