import torch
from typing import Dict, List
import numpy as np

class TrainingMetrics:
    """Tracks and computes training metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.loss_sum = 0.0
        self.steps = 0
        self.tokens_seen = 0
        
    def update(self, loss: float, batch_size: int, seq_length: int):
        self.loss_sum += loss
        self.steps += 1
        self.tokens_seen += batch_size * seq_length
        
    @property
    def average_loss(self) -> float:
        return self.loss_sum / max(1, self.steps)
    
    @property
    def perplexity(self) -> float:
        return np.exp(min(self.average_loss, 20))
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            "loss": self.average_loss,
            "perplexity": self.perplexity,
            "tokens_seen": self.tokens_seen,
            "steps": self.steps
        }

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """Compute token-level accuracy."""
    predictions = logits.argmax(dim=-1)
    mask = labels != ignore_index
    correct = (predictions == labels) & mask
    return correct.sum().float() / mask.sum().float()
