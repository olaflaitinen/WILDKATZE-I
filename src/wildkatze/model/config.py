"""
WILDKATZE-I Model Configuration

This module defines the configuration dataclass for the WILDKATZE-I model,
containing all hyperparameters and architectural specifications.

The default configuration represents the 28B parameter model with:
- 48 transformer layers
- 8192 hidden dimension
- 64 attention heads with 8 KV heads (GQA)
- 32768 token context window
- 128000 token vocabulary

Copyright (c) 2026 olaflaitinen. All rights reserved.
Licensed under EUPL v1.2
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
import yaml


@dataclass
class WildkatzeConfig:
    """
    Configuration class for WILDKATZE-I model.
    
    This dataclass contains all configuration parameters required to
    instantiate a WILDKATZE-I model. The default values correspond to
    the 28B parameter configuration.
    
    Attributes:
        vocab_size: Size of the vocabulary (default: 128000)
        hidden_size: Dimension of hidden states (default: 8192)
        intermediate_size: Dimension of FFN intermediate layer (default: 28672)
        num_hidden_layers: Number of transformer layers (default: 48)
        num_attention_heads: Number of attention heads (default: 64)
        num_key_value_heads: Number of KV heads for GQA (default: 8)
        max_position_embeddings: Maximum sequence length (default: 32768)
        rms_norm_eps: Epsilon for RMSNorm (default: 1e-6)
        rope_theta: Base for RoPE frequency computation (default: 10000.0)
        use_flash_attention: Whether to use Flash Attention (default: True)
        attention_bias: Whether to use bias in attention projections (default: False)
        attention_dropout: Dropout rate for attention (default: 0.0)
        hidden_dropout: Dropout rate for hidden states (default: 0.0)
        hidden_act: Activation function (default: "silu")
        initializer_range: Std for weight initialization (default: 0.02)
        use_cache: Whether to use KV cache (default: True)
        pad_token_id: Padding token ID (default: 0)
        bos_token_id: Beginning of sequence token ID (default: 1)
        eos_token_id: End of sequence token ID (default: 2)
        tie_word_embeddings: Whether to tie input/output embeddings (default: False)
        cultural_context_dim: Dimension of cultural context vector (default: 1024)
    """
    
    # Vocabulary and embedding
    vocab_size: int = 128000
    hidden_size: int = 8192
    
    # Architecture
    intermediate_size: int = 28672
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Positional encoding
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    
    # Attention configuration
    use_flash_attention: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    
    # Dropout
    hidden_dropout: float = 0.0
    
    # Activation
    hidden_act: str = "silu"
    
    # Initialization
    initializer_range: float = 0.02
    
    # Caching
    use_cache: bool = True
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Embedding
    tie_word_embeddings: bool = False
    
    # Specialized modules
    cultural_context_dim: int = 1024
    psychographic_heads: int = 4
    
    # Model identification
    model_type: str = "wildkatze"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
            
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
            
        if self.intermediate_size < self.hidden_size:
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) should be at least "
                f"hidden_size ({self.hidden_size})"
            )
    
    @property
    def head_dim(self) -> int:
        """Compute the dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_key_value_groups(self) -> int:
        """Compute the number of query heads per KV head."""
        return self.num_attention_heads // self.num_key_value_heads
    
    @property
    def num_parameters(self) -> int:
        """Estimate the total number of parameters."""
        # Embedding parameters
        embed_params = self.vocab_size * self.hidden_size
        
        # Per-layer parameters
        attention_params = (
            3 * self.hidden_size * self.hidden_size +  # QKV projections (approximate)
            self.hidden_size * self.hidden_size  # Output projection
        )
        ffn_params = (
            3 * self.hidden_size * self.intermediate_size  # Gate, up, down projections
        )
        norm_params = 2 * self.hidden_size  # Two RMSNorm per layer
        
        layer_params = attention_params + ffn_params + norm_params
        total_layer_params = layer_params * self.num_hidden_layers
        
        # Output head parameters
        output_params = self.vocab_size * self.hidden_size
        
        # Final norm
        final_norm_params = self.hidden_size
        
        return embed_params + total_layer_params + output_params + final_norm_params
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "use_flash_attention": self.use_flash_attention,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "hidden_dropout": self.hidden_dropout,
            "hidden_act": self.hidden_act,
            "initializer_range": self.initializer_range,
            "use_cache": self.use_cache,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
            "cultural_context_dim": self.cultural_context_dim,
            "psychographic_heads": self.psychographic_heads,
            "model_type": self.model_type,
        }
    
    def to_json_string(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_yaml_string(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "WildkatzeConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json_file(cls, json_path: str) -> "WildkatzeConfig":
        """Load configuration from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml_file(cls, yaml_path: str) -> "WildkatzeConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_to_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.to_json_string())
    
    def save_to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml_string())
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"WildkatzeConfig(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_hidden_layers={self.num_hidden_layers},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  num_key_value_heads={self.num_key_value_heads},\n"
            f"  max_position_embeddings={self.max_position_embeddings},\n"
            f"  estimated_parameters={self.num_parameters:,}\n"
            f")"
        )


# Predefined configurations for different model sizes
WILDKATZE_7B_CONFIG = WildkatzeConfig(
    vocab_size=128000,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=32768,
)

WILDKATZE_28B_CONFIG = WildkatzeConfig(
    vocab_size=128000,
    hidden_size=8192,
    intermediate_size=28672,
    num_hidden_layers=48,
    num_attention_heads=64,
    num_key_value_heads=8,
    max_position_embeddings=32768,
)

WILDKATZE_70B_CONFIG = WildkatzeConfig(
    vocab_size=128000,
    hidden_size=12288,
    intermediate_size=43008,
    num_hidden_layers=80,
    num_attention_heads=96,
    num_key_value_heads=8,
    max_position_embeddings=32768,
)
