import pytest
from wildkatze.training.data_loader import WildkatzeDataset, create_dataloader

class MockTokenizer:
    def __call__(self, text, **kwargs):
        import torch
        return {
            "input_ids": torch.randint(0, 100, (1, 128)),
            "attention_mask": torch.ones(1, 128)
        }

def test_dataset_creation(tmp_path):
    # Create temp data file
    data_file = tmp_path / "test_data.jsonl"
    data_file.write_text('{"text": "Test sentence for training."}\n')
    
    tokenizer = MockTokenizer()
    dataset = WildkatzeDataset(str(data_file), tokenizer, max_length=128)
    
    assert len(dataset) == 1
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item

def test_dataloader_creation(tmp_path):
    data_file = tmp_path / "test_data.jsonl"
    data_file.write_text('{"text": "Test."}\n')
    
    tokenizer = MockTokenizer()
    dataset = WildkatzeDataset(str(data_file), tokenizer)
    dataloader = create_dataloader(dataset, batch_size=1)
    
    batch = next(iter(dataloader))
    assert batch["input_ids"].shape[0] == 1
