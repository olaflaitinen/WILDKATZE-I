#!/usr/bin/env python3
"""Clean and preprocess training corpus."""

import argparse
import json
import re
from pathlib import Path

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove null bytes
    text = text.replace('\x00', '')
    return text.strip()

def process_file(input_path: str, output_path: str):
    """Process a single file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned = []
    for line in lines:
        try:
            data = json.loads(line)
            if 'text' in data:
                data['text'] = clean_text(data['text'])
                if len(data['text']) >= 10:
                    cleaned.append(data)
        except json.JSONDecodeError:
            continue
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(cleaned)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean corpus")
    parser.add_argument("input", help="Input file")
    parser.add_argument("output", help="Output file")
    args = parser.parse_args()
    process_file(args.input, args.output)
