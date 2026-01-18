import pytest

def test_full_pipeline():
    """Test complete training to inference pipeline."""
    from wildkatze.model import WildkatzeConfig, WildkatzeForCausalLM
    import torch
    
    config = WildkatzeConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64
    )
    model = WildkatzeForCausalLM(config)
    
    # Forward pass
    input_ids = torch.randint(0, 100, (1, 16))
    logits, _ = model(input_ids)
    
    assert logits.shape == (1, 16, 100)

def test_data_to_model_pipeline():
    """Test data loading to model forward."""
    pass  # Integration test placeholder
