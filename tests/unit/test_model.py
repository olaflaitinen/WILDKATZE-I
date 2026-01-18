import pytest
import torch
from wildkatze.model.config import WildkatzeConfig
from wildkatze.model.architecture import WildkatzeModel

def test_model_initialization():
    config = WildkatzeConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128
    )
    model = WildkatzeModel(config)
    assert model is not None
    assert model.embed_tokens.weight.shape == (1000, 64)

def test_forward_pass():
    config = WildkatzeConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128
    )
    model = WildkatzeModel(config)
    
    input_ids = torch.randint(0, 1000, (2, 32))
    outputs = model(input_ids)
    
    assert outputs["last_hidden_state"].shape == (2, 32, 64)

def test_cultural_context_config():
    config = WildkatzeConfig(cultural_context_dim=512)
    assert config.cultural_context_dim == 512
