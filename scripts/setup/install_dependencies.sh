#!/bin/bash
set -e

echo "Installing WILDKATZE-I dependencies..."

# Install Python dependencies
pip install --upgrade pip
pip install -e .[dev,training,inference]

# Install Flash Attention (requires CUDA)
if command -v nvcc &> /dev/null; then
    pip install flash-attn --no-build-isolation
fi

echo "Installation complete."
