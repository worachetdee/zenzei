#!/bin/bash
set -euo pipefail
MODEL_PATH=${1:?"Usage: run_server.sh <model_path>"}
python -m zensei.serving.api_server --model_path ${MODEL_PATH} --host 0.0.0.0 --port 8000
