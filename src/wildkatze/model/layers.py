"""
WILDKATZE-I Layer Implementations

This module implements the core layer components for WILDKATZE-I:
- RMSNorm: Root Mean Square Layer Normalization
- SwiGLU: Swish-Gated Linear Unit activation

These components are used throughout the model architecture for
normalization and non-linear activation.

Copyright (c) 2026 olaflaitinen. All rights reserved.
Licensed under EUPL v1.2
"""

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WildkatzeRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is a simplification of LayerNorm that removes the mean centering
    and only performs scaling based on the root mean square of activations.
    This provides computational efficiency while maintaining training stability.
    
    The normalization is applied as:
        y = x / RMS(x) * weight
        
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        hidden_size: Dimension of the input tensor
        eps: Small constant for numerical stability (default: 1e-6)
    
    References:
        Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
        NeurIPS 2019.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable scale parameter (no bias in RMSNorm)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the RMS normalization.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute variance (RMS^2)
        variance = x.pow(2).mean(-1, keepdim=True)
        # Normalize
        x = x * torch.rsqrt(variance + self.eps)
        return x
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.
        
        Args:
            hidden_states: Input tensor of shape (..., hidden_size)
            
        Returns:
            Normalized tensor of same shape
        """
        # Convert to float32 for numerical stability during normalization
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # Apply normalization
        hidden_states = self._norm(hidden_states)
        
        # Apply learned scale and convert back to original dtype
        hidden_states = hidden_states.to(input_dtype) * self.weight
        
        return hidden_states
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


class WildkatzeSwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    SwiGLU (Swish-Gated Linear Unit) combines the Swish activation with
    a gating mechanism for improved gradient flow and representation learning.
    
    The transformation is computed as:
        SwiGLU(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down
        
    where Swish(x) = x * sigmoid(x)
    
    This implementation uses three linear projections:
    - gate_proj: Projects to intermediate dimension with Swish activation
    - up_proj: Projects to intermediate dimension without activation
    - down_proj: Projects back to hidden dimension
    
    Args:
        config: Model configuration containing hidden_size and intermediate_size
    
    References:
        Shazeer, N. (2020). GLU Variants Improve Transformer.
        arXiv preprint arXiv:2002.05202.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate projection with Swish activation applied before gating
        self.gate_proj = nn.Linear(
            self.hidden_size, 
            self.intermediate_size, 
            bias=False
        )
        
        # Up projection (value path)
        self.up_proj = nn.Linear(
            self.hidden_size, 
            self.intermediate_size, 
            bias=False
        )
        
        # Down projection back to hidden size
        self.down_proj = nn.Linear(
            self.intermediate_size, 
            self.hidden_size, 
            bias=False
        )
        
        # Swish/SiLU activation function
        self.act_fn = nn.SiLU()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation to the input tensor.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Compute gated activation
        # gate = Swish(x @ W_gate)
        gate = self.act_fn(self.gate_proj(hidden_states))
        
        # Compute up projection
        # up = x @ W_up
        up = self.up_proj(hidden_states)
        
        # Element-wise multiplication (gating)
        # intermediate = gate * up
        intermediate = gate * up
        
        # Project back to hidden dimension
        # output = intermediate @ W_down
        output = self.down_proj(intermediate)
        
        return output
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


class WildkatzeDropout(nn.Module):
    """
    Dropout layer with optional scaling during training.
    
    This is a standard dropout implementation but with additional
    options for use during training and inference.
    
    Args:
        p: Dropout probability (default: 0.0)
        inplace: Whether to perform operation in-place (default: False)
    """
    
    def __init__(self, p: float = 0.0, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to the input tensor.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Tensor with dropout applied during training
        """
        if self.p > 0.0 and self.training:
            return F.dropout(hidden_states, p=self.p, training=True, inplace=self.inplace)
        return hidden_states
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return f"p={self.p}, inplace={self.inplace}"


class WildkatzeLinear(nn.Module):
    """
    Linear layer with optional LoRA support.
    
    This is a standard linear layer that can be extended with Low-Rank
    Adaptation (LoRA) for efficient fine-tuning.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias (default: False)
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Main weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
            
        # LoRA parameters (optional, initialized when enabled)
        self.lora_A = None
        self.lora_B = None
        self.lora_scaling = 1.0
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def enable_lora(self, rank: int = 8, alpha: float = 16.0):
        """
        Enable LoRA adaptation for this layer.
        
        Args:
            rank: Rank of the low-rank matrices
            alpha: Scaling factor for LoRA weights
        """
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_scaling = alpha / rank
        
        # Initialize LoRA A with Kaiming, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation with optional LoRA.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Transformed tensor
        """
        # Standard linear transformation
        output = F.linear(hidden_states, self.weight, self.bias)
        
        # Add LoRA contribution if enabled
        if self.lora_A is not None and self.lora_B is not None:
            lora_output = F.linear(
                F.linear(hidden_states, self.lora_A),
                self.lora_B
            )
            output = output + self.lora_scaling * lora_output
            
        return output
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
