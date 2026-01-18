#!/bin/bash
set -e

# Configuration
MODEL_SIZE="28b"
NUM_GPUS=8
BATCH_SIZE=32
LEARNING_RATE=1e-5

echo "Starting WILDKATZE-I Pretraining..."
echo "Configuration: Model=$MODEL_SIZE, GPUs=$NUM_GPUS, Batch=$BATCH_SIZE"

# Distributed launch
accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    src/wildkatze/training/trainer.py \
    --config configs/training/pretrain.yaml \
    --output_dir models/checkpoints/ \
    --do_train

echo "Training complete."
