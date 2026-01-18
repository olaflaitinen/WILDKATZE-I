#!/usr/bin/env python3
"""Benchmark WILDKATZE-I performance."""

import argparse
import time
import torch
from wildkatze.model import WildkatzeConfig, WildkatzeModel

def benchmark_forward_pass(model, batch_size=1, seq_length=512, iterations=100):
    """Benchmark forward pass latency."""
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(input_ids)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model(input_ids)
    end = time.perf_counter()
    
    avg_latency = (end - start) / iterations * 1000
    return avg_latency

def main():
    parser = argparse.ArgumentParser(description="Benchmark performance")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    
    args = parser.parse_args()
    
    # Create small test model
    config = WildkatzeConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=1024
    )
    model = WildkatzeModel(config)
    
    latency = benchmark_forward_pass(model, args.batch_size, args.seq_length, args.iterations)
    print(f"Average latency: {latency:.2f} ms")

if __name__ == "__main__":
    main()
