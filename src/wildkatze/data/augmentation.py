import random
from typing import List, Dict

class TextAugmenter:
    """Data augmentation for training data."""
    
    def __init__(self, augment_prob: float = 0.1):
        self.augment_prob = augment_prob
        
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms (simplified)."""
        words = text.split()
        for _ in range(min(n, len(words))):
            idx = random.randint(0, len(words) - 1)
            # In production, use a proper synonym database
            words[idx] = words[idx]  # Placeholder
        return ' '.join(words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert n random words."""
        words = text.split()
        for _ in range(n):
            idx = random.randint(0, len(words))
            words.insert(idx, "[MASK]")
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Swap n pairs of words."""
        words = text.split()
        for _ in range(n):
            if len(words) >= 2:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
        return ' '.join(words)
    
    def augment(self, text: str) -> str:
        """Apply random augmentation."""
        if random.random() > self.augment_prob:
            return text
            
        augmentation = random.choice([
            self.synonym_replacement,
            self.random_insertion,
            self.random_swap
        ])
        return augmentation(text)
