import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class InferenceRequest:
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    cultural_context: Optional[Dict] = None
    request_id: str = ""

@dataclass
class InferenceResult:
    text: str
    request_id: str
    tokens_generated: int
    latency_ms: float

class BatchProcessor:
    """Handles dynamic batching for efficient inference."""
    
    def __init__(self, max_batch_size: int = 32, max_wait_time_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.pending_requests: List[InferenceRequest] = []
        self._lock = asyncio.Lock()
        
    async def add_request(self, request: InferenceRequest) -> InferenceResult:
        """Add a request to the batch queue."""
        async with self._lock:
            self.pending_requests.append(request)
            
        # Wait for batch to fill or timeout
        await asyncio.sleep(self.max_wait_time_ms / 1000)
        
        # Process batch (mock implementation)
        return InferenceResult(
            text=f"Response to: {request.prompt}",
            request_id=request.request_id,
            tokens_generated=50,
            latency_ms=25.0
        )
    
    def get_batch(self) -> List[InferenceRequest]:
        """Get the current batch of requests."""
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        return batch
