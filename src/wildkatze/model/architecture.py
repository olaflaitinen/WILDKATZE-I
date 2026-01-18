"""
WILDKATZE-I Model Architecture Implementation

This module implements the complete WILDKATZE-I transformer decoder architecture
with 28 billion parameters, including specialized components for cultural context
processing and psychographic analysis.

Architecture specifications:
- Hidden dimension: 8192
- Intermediate dimension: 28672
- Number of layers: 48
- Attention heads: 64 (with 8 KV heads for GQA)
- Context window: 32768 tokens
- Vocabulary: 128000 tokens

Copyright (c) 2026 olaflaitinen. All rights reserved.
Licensed under EUPL v1.2
"""

from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WildkatzeConfig
from .attention import WildkatzeAttention
from .layers import WildkatzeRMSNorm, WildkatzeSwiGLU
from .embedding import WildkatzeRotaryEmbedding

logger = logging.getLogger(__name__)


class WildkatzeDecoderLayer(nn.Module):
    """
    Single decoder layer implementing the WILDKATZE-I architecture.
    
    Each layer consists of:
    1. Pre-normalization with RMSNorm
    2. Multi-head self-attention with GQA
    3. Residual connection
    4. Pre-normalization with RMSNorm
    5. Feed-forward network with SwiGLU activation
    6. Residual connection
    
    Args:
        config: Model configuration object
        layer_idx: Index of this layer in the decoder stack
    """
    
    def __init__(self, config: WildkatzeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Pre-attention normalization
        self.input_layernorm = WildkatzeRMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        
        # Self-attention mechanism
        self.self_attn = WildkatzeAttention(config, layer_idx)
        
        # Post-attention normalization
        self.post_attention_layernorm = WildkatzeRMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        
        # Feed-forward network
        self.mlp = WildkatzeSwiGLU(config)
        
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
        Forward pass through the decoder layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE
            past_key_value: Cached key-value pairs for autoregressive generation
            cos: Precomputed cosine for RoPE
            sin: Precomputed sine for RoPE
            use_cache: Whether to return key-value cache
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple containing:
            - hidden_states: Output tensor
            - present_key_value: Updated key-value cache (if use_cache)
            - attention_weights: Attention weights (if output_attentions)
        """
        residual = hidden_states
        
        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        hidden_states, present_key_value, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos=cos,
            sin=sin,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Pre-FFN normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Feed-forward network
        hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (present_key_value,)
            
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs


class CulturalContextAdapter(nn.Module):
    """
    Adapter module for integrating cultural context information into the model.
    
    This module allows the model to condition its outputs on cultural context
    vectors that encode information about target culture values, communication
    styles, taboos, and other cultural dimensions.
    
    Args:
        hidden_size: Model hidden dimension
        cultural_dim: Dimension of cultural context vectors (default: 1024)
    """
    
    def __init__(self, hidden_size: int, cultural_dim: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.cultural_dim = cultural_dim
        
        # Project cultural context to hidden dimension
        self.cultural_projection = nn.Linear(cultural_dim, hidden_size, bias=False)
        
        # Gating mechanism for controlled integration
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # Layer normalization for stability
        self.layer_norm = WildkatzeRMSNorm(hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cultural_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate cultural context into hidden states.
        
        Args:
            hidden_states: Model hidden states (batch_size, seq_len, hidden_size)
            cultural_context: Cultural context vector (batch_size, cultural_dim)
            
        Returns:
            Adapted hidden states with cultural context integrated
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project cultural context to hidden dimension
        cultural_embed = self.cultural_projection(cultural_context)
        
        # Expand to sequence length
        cultural_embed = cultural_embed.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute gating values
        combined = torch.cat([hidden_states, cultural_embed], dim=-1)
        gate_values = torch.sigmoid(self.gate_proj(combined))
        
        # Apply gated addition
        adapted = hidden_states + gate_values * cultural_embed
        
        # Normalize output
        return self.layer_norm(adapted)


class WildkatzeModel(nn.Module):
    """
    The core WILDKATZE-I transformer model.
    
    This class implements the full decoder-only transformer architecture
    with 48 layers, RoPE positional encoding, and specialized modules
    for cultural context processing.
    
    Args:
        config: Model configuration object
    """
    
    def __init__(self, config: WildkatzeConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # Token embedding layer
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Rotary position embedding
        self.rotary_emb = WildkatzeRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            WildkatzeDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.norm = WildkatzeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Cultural context adapter (optional)
        if hasattr(config, 'cultural_context_dim') and config.cultural_context_dim > 0:
            self.cultural_adapter = CulturalContextAdapter(
                config.hidden_size,
                config.cultural_context_dim
            )
        else:
            self.cultural_adapter = None
            
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        """Initialize weights using the configured initializer range."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input embedding layer."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embedding layer."""
        self.embed_tokens = value
        
    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Prepare the attention mask for the model.
        
        Creates a causal mask combined with any provided attention mask.
        """
        batch_size, seq_length = input_shape
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=dtype, device=attention_mask.device),
            diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        # Expand for past key values
        if past_key_values_length > 0:
            causal_mask = torch.cat([
                torch.zeros(seq_length, past_key_values_length, dtype=dtype, device=attention_mask.device),
                causal_mask
            ], dim=-1)
            
        # Combine with attention mask
        if attention_mask is not None:
            # Expand attention mask
            expanded_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, seq_length + past_key_values_length
            )
            inverted_mask = 1.0 - expanded_mask
            causal_mask = causal_mask[None, None, :, :] + inverted_mask.masked_fill(
                inverted_mask.bool(), float('-inf')
            )
            
        return causal_mask
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cultural_context: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the transformer model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs for RoPE
            past_key_values: Cached key-value pairs
            inputs_embeds: Pre-computed input embeddings
            cultural_context: Cultural context vector
            use_cache: Whether to use/return key-value cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dict (always True)
            
        Returns:
            Dictionary containing:
            - last_hidden_state: Final hidden states
            - past_key_values: Key-value cache
            - hidden_states: All hidden states (if requested)
            - attentions: Attention weights (if requested)
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        batch_size, seq_length, _ = inputs_embeds.shape
        
        # Past key values length
        past_key_values_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]
            
        # Position IDs
        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                dtype=torch.bool,
                device=inputs_embeds.device
            )
            
        # Get rotary embeddings
        cos, sin = self.rotary_emb(seq_length + past_key_values_length)
        
        hidden_states = inputs_embeds
        
        # Initialize caches
        next_decoder_cache = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # Process through decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    cos,
                    sin,
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    cos=cos,
                    sin=sin,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
                
            if output_attentions:
                all_self_attns += (layer_outputs[-1],)
                
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Apply cultural context adapter if available and context provided
        if self.cultural_adapter is not None and cultural_context is not None:
            hidden_states = self.cultural_adapter(hidden_states, cultural_context)
            
        # Add final hidden states to collection
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


class WildkatzeForCausalLM(nn.Module):
    """
    WILDKATZE-I model with a causal language modeling head.
    
    This class wraps the base WildkatzeModel and adds a linear projection
    to vocabulary logits for next-token prediction.
    
    Args:
        config: Model configuration object
    """
    
    def __init__(self, config: WildkatzeConfig):
        super().__init__()
        self.config = config
        
        # Base transformer model
        self.model = WildkatzeModel(config)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights(self.lm_head)
        
    def _init_weights(self, module: nn.Module):
        """Initialize weights for the LM head."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            
    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input embedding layer."""
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embedding layer."""
        self.model.embed_tokens = value
        
    def get_output_embeddings(self) -> nn.Linear:
        """Return the output embedding layer."""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Set the output embedding layer."""
        self.lm_head = new_embeddings
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cultural_context: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional language modeling loss computation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached key-value pairs
            inputs_embeds: Pre-computed embeddings
            labels: Target labels for loss computation
            cultural_context: Cultural context vector
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dict
            
        Returns:
            Tuple of (logits, loss) where loss is computed if labels provided
        """
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cultural_context=cultural_context,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = outputs["last_hidden_state"]
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            loss = loss_fct(shift_logits, shift_labels)
            
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
        cultural_context: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Prompt token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            do_sample: Whether to sample or use greedy decoding
            cultural_context: Cultural context vector
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated token IDs including the prompt
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        
        batch_size = input_ids.shape[0]
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                outputs = self.model(
                    input_ids=input_ids,
                    cultural_context=cultural_context,
                    use_cache=True,
                )
            else:
                outputs = self.model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    cultural_context=cultural_context,
                    use_cache=True,
                )
                
            past_key_values = outputs["past_key_values"]
            hidden_states = outputs["last_hidden_state"]
            
            # Get next token logits
            next_token_logits = self.lm_head(hidden_states[:, -1, :])
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return input_ids
