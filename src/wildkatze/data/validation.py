import json
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates training and inference data."""
    
    def __init__(self, schema_path: str = None):
        self.schema = self._load_schema(schema_path) if schema_path else {}
        
    def _load_schema(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def validate_training_sample(self, sample: Dict) -> Tuple[bool, List[str]]:
        """Validate a training sample."""
        errors = []
        
        # Check required fields
        if "text" not in sample:
            errors.append("Missing 'text' field")
        elif not isinstance(sample["text"], str):
            errors.append("'text' must be a string")
        elif len(sample["text"]) < 10:
            errors.append("'text' too short (min 10 chars)")
            
        # Check for sensitive content markers
        if "###SENSITIVE###" in sample.get("text", ""):
            errors.append("Contains sensitive content marker")
            
        return len(errors) == 0, errors
    
    def validate_inference_request(self, request: Dict) -> Tuple[bool, List[str]]:
        """Validate an inference request."""
        errors = []
        
        if "prompt" not in request:
            errors.append("Missing 'prompt' field")
        if "target_culture" in request and not isinstance(request["target_culture"], str):
            errors.append("'target_culture' must be a string")
            
        return len(errors) == 0, errors
    
    def sanitize(self, text: str) -> str:
        """Remove potentially harmful content."""
        # Remove known harmful patterns
        sanitized = text.replace("###HARMFUL###", "")
        return sanitized.strip()
