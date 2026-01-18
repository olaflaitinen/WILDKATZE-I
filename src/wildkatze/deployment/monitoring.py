from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    'wildkatze_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status']
)

REQUEST_LATENCY = Histogram(
    'wildkatze_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

INFERENCE_LATENCY = Histogram(
    'wildkatze_inference_latency_seconds',
    'Model inference latency in seconds'
)

ACTIVE_REQUESTS = Gauge(
    'wildkatze_active_requests',
    'Number of active requests'
)

MODEL_MEMORY_USAGE = Gauge(
    'wildkatze_model_memory_bytes',
    'Model memory usage in bytes'
)

class MetricsCollector:
    """Collects and exposes Prometheus metrics."""
    
    def record_request(self, endpoint: str, method: str, status: int, latency: float):
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=str(status)).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        
    def record_inference(self, latency: float):
        INFERENCE_LATENCY.observe(latency)
        
    def set_active_requests(self, count: int):
        ACTIVE_REQUESTS.set(count)
        
    def set_memory_usage(self, bytes_used: int):
        MODEL_MEMORY_USAGE.set(bytes_used)
        
    def get_metrics(self) -> bytes:
        return generate_latest()
