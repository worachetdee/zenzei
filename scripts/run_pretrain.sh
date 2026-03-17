#!/bin/bash
set -euo pipefail
STAGE=${1:-1}
NUM_GPUS=${NUM_GPUS:-8}
CONFIG="configs/training/pretrain_stage${STAGE}.yaml"
echo "=== Starting Stage ${STAGE} pretraining on ${NUM_GPUS} GPUs ==="
deepspeed --num_gpus ${NUM_GPUS} -m zensei.training.pretrain --config ${CONFIG}
