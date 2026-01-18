import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import json

class WildkatzeDataset(Dataset):
    """Dataset for WILDKATZE-I training and evaluation."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer,
        max_length: int = 4096,
        is_training: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.data = self._load_data(data_path)
        
    def _load_data(self, path: str) -> List[Dict]:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item.get("text", "")
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
        }

def create_dataloader(
    dataset: WildkatzeDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
