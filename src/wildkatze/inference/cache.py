from typing import Dict, Tuple, Optional
from collections import OrderedDict

class KVCache:
    """Key-Value cache for efficient autoregressive generation."""
    
    def __init__(self, max_batch_size: int, max_seq_length: int, num_layers: int):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.cache: Dict[int, Tuple] = {}
        
    def get(self, layer_idx: int) -> Optional[Tuple]:
        """Get cached KV for a layer."""
        return self.cache.get(layer_idx)
    
    def set(self, layer_idx: int, key: 'torch.Tensor', value: 'torch.Tensor'):
        """Set cached KV for a layer."""
        self.cache[layer_idx] = (key, value)
        
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        
    def get_seq_length(self) -> int:
        """Get current sequence length in cache."""
        if not self.cache:
            return 0
        first_layer = next(iter(self.cache.values()))
        return first_layer[0].shape[2] if first_layer else 0

class PromptCache:
    """LRU cache for prompt embeddings."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        
    def get(self, prompt_hash: str) -> Optional['torch.Tensor']:
        if prompt_hash in self.cache:
            self.cache.move_to_end(prompt_hash)
            return self.cache[prompt_hash]
        return None
    
    def set(self, prompt_hash: str, embeddings: 'torch.Tensor'):
        if prompt_hash in self.cache:
            self.cache.move_to_end(prompt_hash)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[prompt_hash] = embeddings
