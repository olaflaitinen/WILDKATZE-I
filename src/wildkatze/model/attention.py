"""
WILDKATZE-I Attention Implementation

This module implements the attention mechanism for WILDKATZE-I, including:
- Multi-Head Self-Attention with Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE) integration
- Flash Attention 2 support for optimized computation
- Key-Value caching for efficient autoregressive generation

The attention mechanism uses 64 query heads with 8 key-value heads (GQA),
reducing the KV-cache memory requirement by 8x while maintaining quality.

Copyright (c) 2026 olaflaitinen. All rights reserved.
Licensed under EUPL v1.2
"""

from typing import Optional, Tuple
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dimensions for RoPE.
    
    Takes the input tensor and rotates the second half of the last dimension
    to the first half with negation, as required by RoPE.
    
    Args:
        x: Input tensor of shape (..., dim)
        
    Returns:
        Rotated tensor of same shape
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.
    
    RoPE applies a rotation to the query and key vectors based on their
    position in the sequence. This allows the attention mechanism to
    attend based on relative position without explicit position embeddings.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        cos: Precomputed cosine values (seq_len, head_dim)
        sin: Precomputed sine values (seq_len, head_dim)
        position_ids: Position indices (batch, seq_len)
        
    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as input
    """
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    
    # Apply rotation: q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value heads for Grouped Query Attention.
    
    In GQA, fewer KV heads are used than query heads. This function
    repeats the KV heads to match the number of query heads for
    the attention computation.
    
    Args:
        hidden_states: KV tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each KV head
        
    Returns:
        Expanded tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
        
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class WildkatzeAttention(nn.Module):
    """
    Multi-Head Self-Attention with Grouped Query Attention for WILDKATZE-I.
    
    This implementation uses:
    - 64 query heads with dimension 128 each
    - 8 key-value heads (GQA) for 8x memory reduction
    - RoPE for positional encoding
    - Flash Attention 2 when available
    - Key-value caching for efficient generation
    
    Args:
        config: Model configuration
        layer_idx: Index of this layer in the decoder stack
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size} and num_heads={self.num_heads})"
            )
            
        # Projection layers
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            self.hidden_size, 
            bias=config.attention_bias
        )
        
        # Dropout
        self.attention_dropout = config.attention_dropout
        
        # Check for Flash Attention availability
        self.use_flash_attention = config.use_flash_attention and self._check_flash_attention()
        
    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            if hasattr(F, 'scaled_dot_product_attention'):
                return True
        except Exception:
            pass
        return False
        
    def _attention_standard(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout_p: float,
    ) -> torch.Tensor:
        """
        Standard attention implementation.
        
        Args:
            query_states: (batch, num_heads, seq_len, head_dim)
            key_states: (batch, num_heads, seq_len, head_dim)
            value_states: (batch, num_heads, seq_len, head_dim)
            attention_mask: Attention mask
            dropout_p: Dropout probability
            
        Returns:
            Attention output (batch, num_heads, seq_len, head_dim)
        """
        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, 
            key_states.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        if dropout_p > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
            
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output, attn_weights
        
    def _attention_flash(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        """
        Flash Attention implementation using scaled_dot_product_attention.
        
        Args:
            query_states: Query tensor
            key_states: Key tensor
            value_states: Value tensor
            attention_mask: Attention mask
            dropout_p: Dropout probability
            is_causal: Whether to apply causal masking
            
        Returns:
            Attention output
        """
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p if self.training else 0.0,
            is_causal=is_causal and attention_mask is None,
        )
        return attn_output, None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass through the attention layer.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached KV pairs
            cos: Precomputed cosine for RoPE
            sin: Precomputed sine for RoPE
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of:
            - attn_output: Attention output (batch, seq_len, hidden_size)
            - present_key_value: Updated KV cache (if use_cache)
            - attn_weights: Attention weights (if output_attentions)
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        # Apply RoPE
        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
            
        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            
        present_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        dropout_p = self.attention_dropout if self.training else 0.0
        
        if self.use_flash_attention and not output_attentions:
            attn_output, attn_weights = self._attention_flash(
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout_p,
                is_causal=True,
            )
        else:
            attn_output, attn_weights = self._attention_standard(
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout_p,
            )
            
        # Reshape to (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value, attn_weights
