#!/bin/bash
set -euo pipefail
MODEL_PATH=${1:?"Usage: run_eval.sh <model_path>"}
python -m zensei.eval.run_eval --model_path ${MODEL_PATH} --benchmarks jglue,jaquad,perplexity,mmlu_ja --output_dir results/
