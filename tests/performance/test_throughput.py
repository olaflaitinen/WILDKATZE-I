import pytest
import time
import torch

def test_batch_throughput():
    """Test batch processing throughput."""
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
    
    batch_size = 8
    iterations = 10
    
    input_ids = torch.randint(0, 100, (batch_size, 64))
    
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model(input_ids)
    end = time.perf_counter()
    
    samples_per_second = (batch_size * iterations) / (end - start)
    assert samples_per_second > 1, f"Throughput {samples_per_second} too low"
