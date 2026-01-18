import torch
from typing import Literal
import logging

logger = logging.getLogger(__name__)

QuantizationMode = Literal["int8", "int4", "fp16", "bf16"]

def quantize_model(model: torch.nn.Module, mode: QuantizationMode = "int8") -> torch.nn.Module:
    """
    Quantize model for efficient inference.
    
    Args:
        model: Model to quantize
        mode: Quantization mode (int8, int4, fp16, bf16)
        
    Returns:
        Quantized model
    """
    logger.info(f"Quantizing model to {mode}")
    
    if mode == "int8":
        # Use dynamic quantization for INT8
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
        
    elif mode == "int4":
        # INT4 would require bitsandbytes or similar
        logger.warning("INT4 quantization requires bitsandbytes. Falling back to INT8.")
        return quantize_model(model, "int8")
        
    elif mode == "fp16":
        return model.half()
        
    elif mode == "bf16":
        return model.to(torch.bfloat16)
        
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

def estimate_memory_usage(model: torch.nn.Module, mode: QuantizationMode) -> float:
    """Estimate memory usage in GB for a given quantization mode."""
    param_count = sum(p.numel() for p in model.parameters())
    
    bytes_per_param = {
        "int8": 1,
        "int4": 0.5,
        "fp16": 2,
        "bf16": 2,
    }
    
    return (param_count * bytes_per_param.get(mode, 4)) / (1024**3)
