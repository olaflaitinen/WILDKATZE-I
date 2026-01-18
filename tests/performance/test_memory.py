import pytest
import torch

def test_model_memory_usage():
    """Test model memory footprint."""
    from wildkatze.model import WildkatzeConfig, WildkatzeModel
    
    config = WildkatzeConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128
    )
    model = WildkatzeModel(config)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    memory_mb = (param_count * 4) / (1024 * 1024)  # FP32
    
    print(f"Model parameters: {param_count:,}")
    print(f"Memory (FP32): {memory_mb:.2f} MB")
    
    assert param_count < 1e9, "Test model should be small"
