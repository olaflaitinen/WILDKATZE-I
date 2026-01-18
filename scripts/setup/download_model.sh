#!/bin/bash
set -e

MODEL_NAME=${1:-wildkatze-28b}
MODEL_DIR="./models/checkpoints"

echo "Downloading $MODEL_NAME..."
mkdir -p $MODEL_DIR

# Placeholder for model download
# In production, this would download from a secure repository
echo "Model download not available in this release."
echo "Please contact research@wildkatze.mil for model access."
