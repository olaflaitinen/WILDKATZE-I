#!/bin/bash
set -e

echo "Setting up WILDKATZE-I development environment..."

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
./scripts/setup/install_dependencies.sh

# Create necessary directories
mkdir -p models/checkpoints models/exports data/samples logs

# Setup pre-commit hooks
pre-commit install

echo "Environment setup complete. Activate with: source .venv/bin/activate"
