#!/bin/bash
set -e

echo "Starting WILDKATZE-I Fine-tuning..."

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision bf16 \
    src/wildkatze/training/trainer.py \
    --config configs/training/finetune.yaml \
    --output_dir models/finetuned/ \
    --do_train

echo "Fine-tuning complete."
