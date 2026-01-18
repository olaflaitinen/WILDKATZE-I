import torch
from torch.optim import AdamW
from typing import Optional, List

def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    no_decay_params: Optional[List[str]] = None
) -> AdamW:
    """
    Create an AdamW optimizer with weight decay applied to appropriate parameters.
    
    Args:
        model: The model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        beta1: Beta1 for Adam
        beta2: Beta2 for Adam
        eps: Epsilon for numerical stability
        no_decay_params: List of parameter name patterns to exclude from weight decay
        
    Returns:
        Configured AdamW optimizer
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "layernorm", "layer_norm", "ln_"]
    
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_list = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in no_decay_params):
            no_decay_list.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_list, "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )
    
    return optimizer
