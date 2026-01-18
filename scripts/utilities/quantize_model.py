#!/usr/bin/env python3
"""Quantize WILDKATZE-I model for efficient inference."""

import argparse
import torch
from wildkatze.inference.quantization import quantize_model, estimate_memory_usage

def main():
    parser = argparse.ArgumentParser(description="Quantize model")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--output-path", required=True, help="Output path")
    parser.add_argument("--mode", choices=["int8", "int4", "fp16", "bf16"], default="int8")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model = torch.load(args.model_path, map_location="cpu")
    
    print(f"Quantizing to {args.mode}...")
    quantized = quantize_model(model, args.mode)
    
    print(f"Saving to {args.output_path}...")
    torch.save(quantized.state_dict(), args.output_path)
    
    print("Quantization complete.")

if __name__ == "__main__":
    main()
