from typing import Dict
import logging

logger = logging.getLogger(__name__)

class AutoScaler:
    """Manages horizontal scaling based on metrics."""
    
    def __init__(self, min_replicas: int = 1, max_replicas: int = 10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas
        
    def evaluate(self, metrics: Dict) -> int:
        """Evaluate metrics and return target replica count."""
        cpu_usage = metrics.get("cpu_usage", 0.0)
        request_rate = metrics.get("request_rate", 0)
        latency_p99 = metrics.get("latency_p99_ms", 0)
        
        target = self.current_replicas
        
        # Scale up conditions
        if cpu_usage > 0.8 or latency_p99 > 2000:
            target = min(self.max_replicas, self.current_replicas + 2)
        elif cpu_usage > 0.6 or request_rate > 100:
            target = min(self.max_replicas, self.current_replicas + 1)
            
        # Scale down conditions
        elif cpu_usage < 0.3 and request_rate < 20 and latency_p99 < 500:
            target = max(self.min_replicas, self.current_replicas - 1)
            
        if target != self.current_replicas:
            logger.info(f"Scaling from {self.current_replicas} to {target} replicas")
            self.current_replicas = target
            
        return target
    
    def get_status(self) -> Dict:
        return {
            "current_replicas": self.current_replicas,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas
        }
