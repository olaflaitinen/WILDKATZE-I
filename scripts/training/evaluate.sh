#!/bin/bash
set -e

echo "Running WILDKATZE-I Evaluation..."

python -m pytest tests/ -v --benchmark-only
python scripts/utilities/benchmark_performance.py

echo "Evaluation complete."
