"""
WILDKATZE-I Embedding Implementations

This module implements the embedding components for WILDKATZE-I:
- Rotary Position Embeddings (RoPE)
- Token Embeddings
- Positional utilities

RoPE allows the model to encode relative position information directly
into the attention computation without separate positional embeddings.

Copyright (c) 2026 olaflaitinen. All rights reserved.
Licensed under EUPL v1.2
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn


class WildkatzeRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding implementation.
    
    RoPE applies a rotation to query and key vectors based on their
    position in the sequence. This allows the attention mechanism to
    attend based on relative position without explicit position embeddings.
    
    The rotation is applied as:
        q_rotated = q * cos(theta) + rotate_half(q) * sin(theta)
        
    where theta depends on the position and dimension.
    
    Args:
        dim: Dimension of each attention head
        max_position_embeddings: Maximum sequence length (default: 32768)
        base: Base for frequency computation (default: 10000.0)
        
    References:
        Su, J., et al. (2024). RoFormer: Enhanced Transformer with Rotary
        Position Embedding. Neurocomputing.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache for cos and sin values
        self._build_cache(max_position_embeddings)
        
    def _build_cache(self, seq_len: int):
        """
        Build the cosine and sine cache for the given sequence length.
        
        Args:
            seq_len: Sequence length to build cache for
        """
        # Create position indices
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies: outer product of positions and inverse frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Duplicate frequencies for applying to full dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Compute and cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cosine and sine values for the given sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim)
        """
        # Extend cache if necessary
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
            
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


class WildkatzeLinearScalingRotaryEmbedding(WildkatzeRotaryEmbedding):
    """
    RoPE with linear scaling for extended context lengths.
    
    Linear scaling divides the position by a scaling factor, allowing
    the model to extrapolate to longer sequences than seen during training.
    
    Args:
        dim: Dimension of each attention head
        max_position_embeddings: Maximum sequence length
        base: Base for frequency computation
        scaling_factor: Factor by which to scale positions
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)
        
    def _build_cache(self, seq_len: int):
        """Build cache with linear scaling applied."""
        # Create scaled position indices
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


class WildkatzeDynamicNTKScalingRotaryEmbedding(WildkatzeRotaryEmbedding):
    """
    RoPE with dynamic NTK-aware scaling.
    
    NTK-aware scaling adjusts the RoPE base dynamically based on the
    sequence length, providing better extrapolation than linear scaling.
    
    Args:
        dim: Dimension of each attention head
        max_position_embeddings: Maximum sequence length
        base: Base for frequency computation
        scaling_factor: Factor for NTK scaling
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)
        
    def _build_cache(self, seq_len: int):
        """Build cache with NTK-aware scaling."""
        # Compute scaling for base
        base = self.base * (
            (self.scaling_factor * seq_len / self.max_position_embeddings) 
            - (self.scaling_factor - 1)
        ) ** (self.dim / (self.dim - 2))
        
        # Recompute inverse frequencies with scaled base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim)
        )
        
        # Create position indices
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


def create_rotary_embedding(
    dim: int,
    max_position_embeddings: int,
    base: float = 10000.0,
    scaling_config: Optional[dict] = None,
) -> WildkatzeRotaryEmbedding:
    """
    Factory function to create the appropriate rotary embedding.
    
    Args:
        dim: Dimension of each attention head
        max_position_embeddings: Maximum sequence length
        base: Base for frequency computation
        scaling_config: Optional dict with 'type' and 'factor' keys
        
    Returns:
        Appropriate rotary embedding instance
    """
    if scaling_config is None:
        return WildkatzeRotaryEmbedding(dim, max_position_embeddings, base)
        
    scaling_type = scaling_config.get("type", "linear")
    scaling_factor = scaling_config.get("factor", 1.0)
    
    if scaling_type == "linear":
        return WildkatzeLinearScalingRotaryEmbedding(
            dim, max_position_embeddings, base, scaling_factor
        )
    elif scaling_type == "dynamic":
        return WildkatzeDynamicNTKScalingRotaryEmbedding(
            dim, max_position_embeddings, base, scaling_factor
        )
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}")


class WildkatzeTokenEmbedding(nn.Module):
    """
    Token embedding layer with optional position-aware initialization.
    
    This embedding layer converts discrete token IDs to continuous
    vectors that serve as input to the transformer.
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of the embedding vectors
        padding_idx: Optional token ID to use for padding (zeros)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        
        # Embedding matrix
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
        )
        
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Convert token IDs to embedding vectors.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            
        Returns:
            Embedding vectors of shape (batch, seq_len, hidden_size)
        """
        return self.embedding(input_ids)
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (
            f"vocab_size={self.vocab_size}, "
            f"hidden_size={self.hidden_size}, "
            f"padding_idx={self.padding_idx}"
        )
