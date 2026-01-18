import pytest
import time
import torch

def test_forward_latency():
    """Test forward pass latency meets requirements (<2s)."""
    from wildkatze.model import WildkatzeConfig, WildkatzeModel
    
    config = WildkatzeConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128
    )
    model = WildkatzeModel(config)
    model.eval()
    
    input_ids = torch.randint(0, 100, (1, 64))
    
    start = time.perf_counter()
    with torch.no_grad():
        model(input_ids)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    assert latency_ms < 2000, f"Latency {latency_ms}ms exceeds 2s requirement"
