import pytest

def test_inference_engine_initialization():
    from wildkatze.inference.engine import WildkatzeInferenceEngine
    engine = WildkatzeInferenceEngine(model_path="./models/test")
    assert engine is not None

def test_batch_processor():
    from wildkatze.inference.batch_processor import BatchProcessor, InferenceRequest
    
    processor = BatchProcessor(max_batch_size=8)
    request = InferenceRequest(prompt="Test", request_id="001")
    
    assert processor.max_batch_size == 8
    assert request.prompt == "Test"

def test_kv_cache():
    from wildkatze.inference.cache import KVCache
    
    cache = KVCache(max_batch_size=4, max_seq_length=1024, num_layers=32)
    assert cache.get_seq_length() == 0
    cache.clear()
