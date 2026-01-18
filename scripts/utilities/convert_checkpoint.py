#!/usr/bin/env python3
"""Convert model checkpoints between formats."""

import argparse
import torch
from safetensors.torch import save_file, load_file

def convert_pytorch_to_safetensors(input_path: str, output_path: str):
    """Convert PyTorch checkpoint to SafeTensors format."""
    state_dict = torch.load(input_path, map_location="cpu")
    save_file(state_dict, output_path)
    print(f"Converted {input_path} -> {output_path}")

def convert_safetensors_to_pytorch(input_path: str, output_path: str):
    """Convert SafeTensors to PyTorch checkpoint."""
    state_dict = load_file(input_path)
    torch.save(state_dict, output_path)
    print(f"Converted {input_path} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model checkpoints")
    parser.add_argument("input", help="Input checkpoint path")
    parser.add_argument("output", help="Output checkpoint path")
    parser.add_argument("--to-safetensors", action="store_true")
    parser.add_argument("--to-pytorch", action="store_true")
    
    args = parser.parse_args()
    
    if args.to_safetensors:
        convert_pytorch_to_safetensors(args.input, args.output)
    elif args.to_pytorch:
        convert_safetensors_to_pytorch(args.input, args.output)
